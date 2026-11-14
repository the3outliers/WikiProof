"""Forensic LLM diff analyzer using Hostinger AI Router with Gemini fallback.

Sends the MediaWiki revision diff (+ added / - deleted text), behavioral risk
signals, and false-positive guardrails to an OpenAI-compatible LLM endpoint
(Hostinger AI Router) or falls back automatically to the Google GenAI SDK.
Enforces strict JSON output matching WikiProof's forensic verdict schema.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from config import get_logger, settings

logger = get_logger("wikiproof.llm_analyzer")

ALLOWED_VERDICTS = {"Suspicious", "Uncertain", "Benign"}
ALLOWED_TECHNIQUES = [
    "Whitewashing",
    "Loaded Framing",
    "Fabricated Authority",
    "Source Laundering",
]

SYSTEM_PROMPT = """You are WikiProof, an impartial Wikipedia forensic threat-intelligence analyst.
Evaluate whether a Wikipedia article revision represents covert manipulation, coordinated influence,
conflict-of-interest editing, or vandalism versus legitimate volunteer editing.

You MUST account for BOTH behavioral risk signals AND false-positive guardrails.
- Veteran editors removing unverified claims or copyediting sensitive articles are often Benign.
- Adding peer-reviewed, wire-service, or official academic citations is strong evidence of Benign editing.
- If the diff is ambiguous, minor, or lacks sufficient evidence of bad-faith manipulation, choose "Uncertain" rather than over-flagging.

Return STRICT JSON ONLY with these exact keys:
{
  "verdict": "Suspicious" | "Uncertain" | "Benign",
  "confidence": <float between 0.0 and 1.0>,
  "techniques": [<subset of ["Whitewashing", "Loaded Framing", "Fabricated Authority", "Source Laundering"]; MUST be [] if verdict is "Benign">],
  "highlighted_quote": "<exact verbatim substring copied from the ADDED or DELETED diff text that most strongly drives your verdict, or empty string if trivial>",
  "explanation": "<exactly 2 clear, plain-English sentences explaining the verdict to an everyday reader>"
}

Technique definitions:
- "Whitewashing": Removing sourced criticism, controversies, negative history, or unfavorable facts without neutral replacement.
- "Loaded Framing": Injecting biased, promotional, peacock, or emotionally slanted wording presented as encyclopedic fact.
- "Fabricated Authority": Asserting sweeping, extraordinary, or self-serving claims without reliable citations or with bogus/misrepresented credentials.
- "Source Laundering": Injecting low-reliability, partisan, self-published, or promotional external domains to create an illusion of verification."""


def _truncate_text(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _build_user_prompt(
    revision: dict[str, Any],
    user_profile: dict[str, Any],
    heuristics: dict[str, Any],
) -> str:
    added_text = _truncate_text(revision.get("added_text") or "(none)", 3200)
    deleted_text = _truncate_text(revision.get("deleted_text") or "(none)", 3200)

    signals = [
        f"- {s['name']} (+{s['weight']}): {s['detail']}"
        for s in heuristics.get("triggered_signals", [])
    ]
    guardrails = [
        f"- {g['name']} ({g['adjustment']}): {g['detail']}"
        for g in heuristics.get("guardrails_applied", [])
    ]

    payload = {
        "article_title": revision.get("title"),
        "wiki": revision.get("wiki"),
        "wiki_domain": revision.get("wiki_domain"),
        "rev_id": revision.get("rev_id"),
        "edit_summary": revision.get("comment") or "(empty)",
        "editor": {
            "username": user_profile.get("username"),
            "age_days": user_profile.get("age_days"),
            "editcount": user_profile.get("editcount"),
            "groups": user_profile.get("groups", []),
            "is_anonymous": user_profile.get("is_anonymous", False),
            "is_temp": user_profile.get("is_temp", False),
        },
        "heuristics": {
            "score": heuristics.get("score"),
            "risk_tier": heuristics.get("risk_tier"),
            "added_domains": heuristics.get("added_domains", []),
            "verified_domains": heuristics.get("verified_domains", []),
            "unverified_domains": heuristics.get("unverified_domains", []),
        },
    }

    return (
        f"CONTEXT JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"TRIGGERED RISK SIGNALS:\n"
        f"{chr(10).join(signals) if signals else '- None'}\n\n"
        f"FALSE-POSITIVE GUARDRAILS APPLIED:\n"
        f"{chr(10).join(guardrails) if guardrails else '- None'}\n\n"
        f"DIFF - DELETED TEXT (-):\n{deleted_text}\n\n"
        f"DIFF - ADDED TEXT (+):\n{added_text}\n"
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON object from model output, stripping markdown code fences if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a valid JSON object")


def _select_verbatim_quote(candidate: str, revision: dict[str, Any]) -> str:
    """Ensure highlighted_quote is an actual substring of the diff when possible."""
    added_text = revision.get("added_text") or ""
    deleted_text = revision.get("deleted_text") or ""
    combined = f"{added_text}\n{deleted_text}"

    clean_cand = (candidate or "").strip().strip('"').strip("'")
    if clean_cand and clean_cand in combined:
        return clean_cand

    # Try matching first sentence/clause or fallback to actual diff highlight
    if clean_cand and len(clean_cand) > 20:
        snippet = clean_cand[:40]
        idx = combined.find(snippet)
        if idx != -1:
            return combined[idx : min(len(combined), idx + 180)].splitlines()[0]

    highlights = revision.get("added_highlights") or revision.get("deleted_highlights") or []
    for h in highlights:
        if len(h.strip()) >= 8:
            return h.strip()[:220]

    lines = revision.get("added_lines") or revision.get("deleted_lines") or []
    for line in lines:
        if line.strip():
            return line.strip()[:220]

    return ""


def _normalize_llm_output(
    raw_obj: dict[str, Any],
    revision: dict[str, Any],
    provider: str,
    model: str,
    latency_ms: int,
) -> dict[str, Any]:
    verdict = str(raw_obj.get("verdict", "Uncertain")).strip()
    if verdict not in ALLOWED_VERDICTS:
        verdict = "Uncertain"

    try:
        confidence = float(raw_obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    raw_techniques = raw_obj.get("techniques") or []
    if not isinstance(raw_techniques, list):
        raw_techniques = []
    techniques = [t for t in raw_techniques if t in ALLOWED_TECHNIQUES]
    # Deduplicate preserving order
    techniques = list(dict.fromkeys(techniques))
    if verdict == "Benign":
        techniques = []

    quote = _select_verbatim_quote(str(raw_obj.get("highlighted_quote") or ""), revision)
    explanation = str(raw_obj.get("explanation") or "").strip()
    if not explanation:
        explanation = (
            f"Automated forensic analysis classified this edit to '{revision.get('title')}' as {verdict}. "
            f"The assessment reflects both behavioral account/link signals and the text changed in the diff."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "techniques": techniques,
        "highlighted_quote": quote,
        "explanation": explanation,
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
    }


def _heuristic_abstention_fallback(
    revision: dict[str, Any],
    heuristics: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Deterministic fallback when no external LLM provider is reachable/configured."""
    score = int(heuristics.get("score") or 0)
    signal_ids = {s["id"] for s in heuristics.get("triggered_signals", [])}

    if score >= settings.risk_threshold_suspicious:
        verdict = "Suspicious"
        confidence = round(min(0.92, 0.55 + score / 250.0), 3)
    elif score >= settings.risk_threshold_uncertain:
        verdict = "Uncertain"
        confidence = round(0.45 + (score - 30) / 150.0, 3)
    else:
        verdict = "Benign"
        confidence = round(max(0.60, 0.92 - score / 100.0), 3)

    techniques: list[str] = []
    if verdict != "Benign":
        if "large_content_or_citation_deletion" in signal_ids:
            techniques.append("Whitewashing")
        if "unverified_external_link" in signal_ids:
            techniques.append("Source Laundering")
        if "promotional_peacock_framing" in signal_ids or "coi_username_article_overlap" in signal_ids:
            techniques.append("Loaded Framing")
        if "unsourced_substantive_addition" in signal_ids and not techniques:
            techniques.append("Fabricated Authority")

    quote = _select_verbatim_quote("", revision)
    signal_names = [s["name"] for s in heuristics.get("triggered_signals", [])]
    guardrail_names = [g["name"] for g in heuristics.get("guardrails_applied", [])]

    if verdict == "Suspicious":
        s1 = (
            f"Behavioral heuristics flagged this revision on '{revision.get('title')}' with a risk score of {score}/100 "
            f"due to {', '.join(signal_names[:2]) or 'elevated risk indicators'}."
        )
        s2 = "External LLM analysis was unavailable, so this verdict relies on deterministic diff and editor profile signals."
    elif verdict == "Benign":
        s1 = (
            f"This edit to '{revision.get('title')}' scored {score}/100 and appears consistent with normal volunteer editing"
            + (f" ({', '.join(guardrail_names[:2])})." if guardrail_names else ".")
        )
        s2 = "No unverified external link injection or unexplained citation removal was detected in the revision."
    else:
        s1 = (
            f"This revision scored {score}/100 with mixed behavioral indicators, placing it in the uncertain review band."
        )
        s2 = f"Human review is recommended because external AI adjudication could not run ({reason})."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "techniques": techniques,
        "highlighted_quote": quote,
        "explanation": f"{s1} {s2}",
        "provider": "heuristic_fallback",
        "model": "deterministic-heuristics-v1",
        "latency_ms": 0,
    }


async def _call_hostinger_router(user_prompt: str) -> dict[str, Any]:
    """Call Hostinger AI Router via OpenAI-compatible SDK."""
    logger.info(
        "LLM request provider=hostinger_ai_router base_url=%s model=%s prompt_chars=%d",
        settings.hostinger_base_url,
        settings.hostinger_model_slug,
        len(user_prompt),
    )
    client = AsyncOpenAI(
        api_key=settings.hostinger_ai_key,
        base_url=settings.hostinger_base_url,
        timeout=25.0,
    )
    try:
        response = await client.chat.completions.create(
            model=settings.hostinger_model_slug,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return _extract_json_object(content)
    finally:
        await client.close()


async def _call_gemini_fallback(user_prompt: str) -> dict[str, Any]:
    """Fallback to Google GenAI SDK using GEMINI_API_KEY and GEMINI_MODEL_SLUG."""
    logger.info(
        "LLM request provider=gemini_fallback model=%s prompt_chars=%d",
        settings.gemini_model_slug,
        len(user_prompt),
    )

    def _sync_call() -> str:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model_slug,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "verdict": {"type": "STRING", "enum": ["Suspicious", "Uncertain", "Benign"]},
                        "confidence": {"type": "NUMBER"},
                        "techniques": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                        "highlighted_quote": {"type": "STRING"},
                        "explanation": {"type": "STRING"},
                    },
                    "required": [
                        "verdict",
                        "confidence",
                        "techniques",
                        "highlighted_quote",
                        "explanation",
                    ],
                },
            ),
        )
        return response.text or ""

    raw_text = await asyncio.to_thread(_sync_call)
    return _extract_json_object(raw_text)


async def analyze_revision(
    revision: dict[str, Any],
    user_profile: dict[str, Any],
    heuristics: dict[str, Any],
) -> dict[str, Any]:
    """Analyze a revision diff with Hostinger AI Router and Gemini SDK fallback."""
    user_prompt = _build_user_prompt(revision, user_profile, heuristics)
    start = time.perf_counter()

    # Primary: Hostinger AI Router
    if settings.has_valid_hostinger_key():
        try:
            raw_obj = await _call_hostinger_router(user_prompt)
            latency_ms = int((time.perf_counter() - start) * 1000)
            result = _normalize_llm_output(
                raw_obj,
                revision,
                provider="hostinger_ai_router",
                model=settings.hostinger_model_slug,
                latency_ms=latency_ms,
            )
            logger.info(
                "LLM response provider=hostinger_ai_router rev_id=%s verdict=%s confidence=%.2f latency_ms=%d",
                revision.get("rev_id"),
                result["verdict"],
                result["confidence"],
                latency_ms,
            )
            return result
        except Exception as exc:
            logger.warning("Hostinger AI Router call failed (%s); attempting Gemini fallback", exc)
    else:
        logger.info("HOSTINGER_AI_KEY not configured; attempting Gemini fallback")

    # Secondary: Google GenAI SDK
    if settings.has_valid_gemini_key():
        start_gem = time.perf_counter()
        try:
            raw_obj = await _call_gemini_fallback(user_prompt)
            latency_ms = int((time.perf_counter() - start_gem) * 1000)
            result = _normalize_llm_output(
                raw_obj,
                revision,
                provider="gemini_fallback",
                model=settings.gemini_model_slug,
                latency_ms=latency_ms,
            )
            logger.info(
                "LLM response provider=gemini_fallback rev_id=%s verdict=%s confidence=%.2f latency_ms=%d",
                revision.get("rev_id"),
                result["verdict"],
                result["confidence"],
                latency_ms,
            )
            return result
        except Exception as exc:
            logger.warning("Gemini fallback call failed (%s); using heuristic abstention fallback", exc)
            return _heuristic_abstention_fallback(revision, heuristics, f"LLM error: {type(exc).__name__}")

    logger.warning("Neither HOSTINGER_AI_KEY nor GEMINI_API_KEY is configured; using heuristic fallback")
    return _heuristic_abstention_fallback(revision, heuristics, "no LLM API key configured")
