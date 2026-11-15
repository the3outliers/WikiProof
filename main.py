"""FastAPI backend and Wikimedia SSE worker for WikiProof.

Provides REST endpoints for live scored edit telemetry, on-demand forensic
revision inspection, real preset examples (True Positive, Hard Negative,
Uncertain), Editor -> Article -> External Domain network graphs, and
threshold-adjustable benchmark evaluation.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import get_logger, settings
from database import (
    find_cross_wiki_bursts,
    get_benchmark_report,
    get_cached_analysis_inputs,
    get_cached_revision,
    get_db_stats,
    get_live_feed,
    get_propaganda_flagged,
    get_network_graph,
    get_presets,
    init_db,
    save_analyzed_revision,
    save_cached_diff_summary,
    save_live_event,
)
from heuristics import score_revision
from topics import changed_segments, detect_sensitive_topics
from llm_analyzer import analyze_revision
from wikimedia_client import (
    fetch_revision_details,
    fetch_user_profile,
    parse_revision_input,
    stream_recent_changes,
    wiki_to_domain,
)

logger = get_logger("wikiproof.main")


class StreamState:
    """Tracks background SSE ingestion worker health and counters."""

    def __init__(self) -> None:
        self.active: bool = False
        self.events_received: int = 0
        self.events_processed: int = 0
        self.last_event_at: str | None = None
        self.last_error: str | None = None
        self.stop_event: asyncio.Event = asyncio.Event()
        self.task: asyncio.Task[None] | None = None


stream_state = StreamState()


class InspectRequest(BaseModel):
    """Request payload for POST /api/inspect."""

    url: str | None = Field(default=None, description="Wikipedia diff or oldid URL")
    wiki_domain: str | None = Field(default=None, description="e.g. en.wikipedia.org")
    rev_id: int | None = Field(default=None, description="MediaWiki revision ID")
    force_refresh: bool = Field(default=False, description="Bypass SQLite cache and re-analyze")


class EnglishSummaryRequest(BaseModel):
    """Request payload for generating an English summary of a cached revision diff."""

    wiki_domain: str = Field(description="Wikipedia domain, e.g. sv.wikipedia.org")
    rev_id: int = Field(gt=0, description="MediaWiki revision ID")


async def run_full_forensic_pipeline(
    wiki_domain: str,
    rev_id: int,
    source: str = "inspect",
    preset_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch real diff & profile, compute heuristics + bursts, run LLM, and cache."""
    revision = await fetch_revision_details(wiki_domain, rev_id, client=client)
    user_profile = await fetch_user_profile(wiki_domain, revision["user"], client=client)
    cross_wiki = find_cross_wiki_bursts(
        title=revision["title"],
        added_domains=revision["added_domains"],
        current_wiki=revision["wiki"],
    )
    heuristics = score_revision(revision, user_profile, cross_wiki_matches=cross_wiki)
    llm_analysis = await analyze_revision(revision, user_profile, heuristics)
    dossier = save_analyzed_revision(
        revision=revision,
        user_profile=user_profile,
        heuristics=heuristics,
        llm_analysis=llm_analysis,
        source=source,
        preset_key=preset_key,
    )
    return dossier


async def _sse_ingestion_worker() -> None:
    """Background worker consuming Wikimedia EventStreams SSE and scoring edits."""
    stream_state.active = True
    last_llm_ts = 0.0
    last_topic_llm_ts = 0.0
    semaphore = asyncio.Semaphore(2)
    pending_tasks: set[asyncio.Task[None]] = set()
    MAX_PENDING_EVENTS = 20

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        async for event in stream_recent_changes(stop_event=stream_state.stop_event):
            stream_state.events_received += 1
            stream_state.last_event_at = datetime.now(timezone.utc).isoformat()

            rev_obj = event.get("revision") or {}
            rev_id = rev_obj.get("new")
            wiki = event.get("wiki")
            if not rev_id or not wiki:
                continue

            wiki_domain = wiki_to_domain(wiki)

            async def _process_one(w_domain: str, r_id: int) -> None:
                nonlocal last_llm_ts, last_topic_llm_ts
                async with semaphore:
                    try:
                        revision = await fetch_revision_details(w_domain, r_id, client=client)
                        user_profile = await fetch_user_profile(w_domain, revision["user"], client=client)
                        cross_wiki = find_cross_wiki_bursts(
                            title=revision["title"],
                            added_domains=revision["added_domains"],
                            current_wiki=revision["wiki"],
                        )
                        heuristics = score_revision(revision, user_profile, cross_wiki_matches=cross_wiki)

                        llm_analysis: dict[str, Any] | None = None
                        now_mono = time.monotonic()
                        run_llm = False
                        if (
                            heuristics["score"] >= settings.llm_auto_analyze_threshold
                            and (now_mono - last_llm_ts) >= 12.0
                        ):
                            last_llm_ts = now_mono
                            run_llm = True
                        elif (
                            settings.sensitive_topic_llm_enabled
                            and (now_mono - last_topic_llm_ts) >= settings.sensitive_topic_llm_interval_seconds
                        ):
                            changed = changed_segments(revision)
                            topics = (
                                detect_sensitive_topics(revision.get("title", ""), changed)
                                if sum(len(s) for s in changed) >= settings.sensitive_topic_min_changed_chars
                                else {}
                            )
                            if topics:
                                last_topic_llm_ts = now_mono
                                run_llm = True
                                revision["sensitive_topics"] = topics
                                logger.info(
                                    "Sensitive-topic AI review %s/%s '%s' (score %s): %s",
                                    w_domain, r_id, revision.get("title"), heuristics["score"], topics,
                                )

                        if run_llm:
                            llm_analysis = await analyze_revision(revision, user_profile, heuristics)
                            save_analyzed_revision(
                                revision=revision,
                                user_profile=user_profile,
                                heuristics=heuristics,
                                llm_analysis=llm_analysis,
                                source="stream",
                            )

                        save_live_event(revision=revision, heuristics=heuristics, llm_analysis=llm_analysis)
                        stream_state.events_processed += 1
                    except Exception as exc:
                        stream_state.last_error = str(exc)
                        logger.debug("Live event processing skipped for %s/%s: %s", w_domain, r_id, exc)

            # Fire-and-forget bounded task so SSE stream never blocks. When the
            # backlog is full (e.g. while backing off a Wikipedia 429), sample the
            # firehose instead of queueing unbounded work.
            if len(pending_tasks) >= MAX_PENDING_EVENTS:
                continue
            task = asyncio.create_task(_process_one(wiki_domain, int(rev_id)))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize SQLite storage and manage the background Wikimedia SSE worker."""
    init_db()
    if settings.stream_enabled:
        stream_state.stop_event.clear()
        stream_state.task = asyncio.create_task(_sse_ingestion_worker())
        logger.info("Started Wikimedia EventStreams SSE background worker")
    yield
    stream_state.stop_event.set()
    stream_state.active = False
    if stream_state.task is not None:
        stream_state.task.cancel()
        try:
            await stream_state.task
        except asyncio.CancelledError:
            pass
    logger.info("Stopped Wikimedia EventStreams SSE background worker")


app = FastAPI(
    title="WikiProof Forensic API",
    description="Real-time Wikimedia SSE ingestion, behavioral heuristics, and LLM forensic diff analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX = Path(__file__).resolve().parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the WikiProof threat-intelligence dashboard directly from FastAPI."""
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND_INDEX)


@app.get("/api/health")
async def healthcheck() -> dict[str, Any]:
    """Return service health, SQLite counts, SSE worker status, and LLM config flags."""
    return {
        "status": "ok",
        "service": "WikiProof",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stream": {
            "enabled": settings.stream_enabled,
            "active": stream_state.active,
            "wikis": sorted(settings.wiki_stream_wikis),
            "events_received": stream_state.events_received,
            "events_processed": stream_state.events_processed,
            "last_event_at": stream_state.last_event_at,
            "last_error": stream_state.last_error,
        },
        "db_stats": get_db_stats(),
        "llm": {
            "hostinger_configured": settings.has_valid_hostinger_key(),
            "hostinger_base_url": settings.hostinger_base_url,
            "hostinger_model_slug": settings.hostinger_model_slug,
            "gemini_fallback_configured": settings.has_valid_gemini_key(),
            "gemini_model_slug": settings.gemini_model_slug,
        },
    }


@app.get("/api/live-feed")
async def live_feed(
    limit: int = Query(default=50, ge=1, le=200),
    wiki: str | None = Query(default=None),
    min_score: int = Query(default=0, ge=0, le=100),
    propaganda: bool = Query(default=False, description="Only edits the AI flagged as propaganda"),
) -> dict[str, Any]:
    """Return latest scored edits ingested from the Wikimedia SSE stream."""
    events = get_live_feed(limit=limit, wiki=wiki, min_score=min_score, propaganda_only=propaganda)
    return {
        "count": len(events),
        "events": events,
    }


@app.get("/api/propaganda")
async def propaganda_flagged(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    """Return every analyzed revision whose AI assessment detected a propaganda narrative."""
    items = get_propaganda_flagged(limit=limit)
    return {"count": len(items), "items": items}


@app.post("/api/inspect")
async def inspect_revision(payload: InspectRequest) -> dict[str, Any]:
    """Inspect a Wikipedia revision URL or (wiki_domain, rev_id) via full forensic pipeline."""
    try:
        wiki_domain, rev_id = parse_revision_input(
            url=payload.url,
            wiki_domain=payload.wiki_domain,
            rev_id=payload.rev_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not payload.force_refresh:
        cached = get_cached_revision(wiki_domain, rev_id)
        if cached is not None:
            return {**cached, "cached": True}

    try:
        dossier = await run_full_forensic_pipeline(wiki_domain, rev_id, source="inspect")
        return {**dossier, "cached": False}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to inspect %s/%s", wiki_domain, rev_id)
        raise HTTPException(status_code=502, detail=f"Upstream analysis error: {exc}") from exc


@app.post("/api/english-summary")
async def generate_english_summary(payload: EnglishSummaryRequest) -> dict[str, str]:
    """Retry AI analysis for a cached revision and persist its English diff summary."""
    inputs = get_cached_analysis_inputs(payload.wiki_domain, payload.rev_id)
    if inputs is None:
        raise HTTPException(status_code=404, detail="Revision is not available in the local analysis cache.")

    existing_summary = inputs["llm_analysis"].get("diff_summary_en")
    if existing_summary:
        return {"diff_summary_en": existing_summary}

    llm_analysis = await analyze_revision(
        inputs["revision"],
        inputs["user_profile"],
        inputs["heuristics"],
    )
    summary = llm_analysis.get("diff_summary_en")
    if not summary:
        raise HTTPException(
            status_code=503,
            detail="AI summary unavailable. Verify HOSTINGER_AI_KEY or configure GEMINI_API_KEY, then retry.",
        )
    if not save_cached_diff_summary(payload.wiki_domain, payload.rev_id, summary):
        raise HTTPException(status_code=404, detail="Revision is no longer available in the local analysis cache.")
    return {"diff_summary_en": summary}


@app.get("/api/presets")
async def presets() -> dict[str, Any]:
    """Return 3 cached real examples: True Positive, Hard-Negative Benign, and Uncertain."""
    items = get_presets()
    if len(items) < 3:
        # Lazy-seed the 3 canonical real revisions if seed_benchmark.py has not been run yet
        default_presets = [
            ("en.wikipedia.org", 1385614788, "true_positive"),
            ("en.wikipedia.org", 1351299528, "hard_negative"),
            ("en.wikipedia.org", 1385616046, "uncertain"),
        ]
        existing_keys = {item.get("preset_key") for item in items}
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            for dom, rid, pkey in default_presets:
                if pkey not in existing_keys:
                    try:
                        await run_full_forensic_pipeline(dom, rid, source="preset", preset_key=pkey, client=client)
                    except Exception as exc:
                        logger.warning("Failed to lazy-seed preset %s (%s/%s): %s", pkey, dom, rid, exc)
        items = get_presets()

    return {
        "count": len(items),
        "presets": items,
    }


@app.get("/api/network")
async def network(
    min_score: int = Query(default=0, ge=0, le=100),
    limit: int = Query(default=120, ge=1, le=500),
) -> dict[str, Any]:
    """Return Editor -> Article -> External Domain graph nodes and edges."""
    return get_network_graph(min_score=min_score, limit=limit)


@app.get("/api/benchmark")
async def benchmark(
    threshold: int = Query(default=settings.risk_threshold_default, ge=0, le=100),
) -> dict[str, Any]:
    """Return stored benchmark results with dynamic confusion matrix recalculation."""
    report = get_benchmark_report(threshold=threshold)
    return report


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )
