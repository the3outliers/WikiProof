"""SQLite persistence layer for WikiProof revisions, live stream, network, and benchmarks.

Manages thread-safe SQLite storage (`data/wikiproof.db`) with WAL mode enabled,
caching inspected revisions, storing Wikimedia SSE live feed events, tracking
cross-language bursts, building Editor -> Article -> External Domain graphs,
and dynamically recalculating benchmark confusion matrices across risk thresholds.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from config import get_logger, settings
from heuristics import normalize_topic

logger = get_logger("wikiproof.database")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(file_path: str) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection configured with Row factory and WAL mode."""
    target = _ensure_parent_dir(db_path or settings.db_path)
    conn = sqlite3.connect(str(target), timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create SQLite schema if not already present."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wiki_domain TEXT NOT NULL,
                rev_id INTEGER NOT NULL,
                parent_id INTEGER DEFAULT 0,
                wiki TEXT NOT NULL,
                title TEXT NOT NULL,
                normalized_topic TEXT NOT NULL,
                user TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                comment TEXT DEFAULT '',
                diff_url TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                added_chars INTEGER DEFAULT 0,
                deleted_chars INTEGER DEFAULT 0,
                added_urls_json TEXT NOT NULL DEFAULT '[]',
                added_domains_json TEXT NOT NULL DEFAULT '[]',
                triggered_signals_json TEXT NOT NULL DEFAULT '[]',
                guardrails_json TEXT NOT NULL DEFAULT '[]',
                user_profile_json TEXT NOT NULL DEFAULT '{}',
                revision_detail_json TEXT NOT NULL DEFAULT '{}',
                llm_analysis_json TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT 'inspect',
                preset_key TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(wiki_domain, rev_id)
            );

            CREATE INDEX IF NOT EXISTS idx_revisions_created ON revisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_revisions_preset ON revisions(preset_key);

            CREATE TABLE IF NOT EXISTS live_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wiki TEXT NOT NULL,
                wiki_domain TEXT NOT NULL,
                rev_id INTEGER NOT NULL,
                parent_id INTEGER DEFAULT 0,
                title TEXT NOT NULL,
                normalized_topic TEXT NOT NULL,
                user TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                comment TEXT DEFAULT '',
                diff_url TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                added_domains_json TEXT NOT NULL DEFAULT '[]',
                triggered_signals_json TEXT NOT NULL DEFAULT '[]',
                guardrails_json TEXT NOT NULL DEFAULT '[]',
                verdict TEXT DEFAULT NULL,
                confidence REAL DEFAULT NULL,
                explanation TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(wiki_domain, rev_id)
            );

            CREATE INDEX IF NOT EXISTS idx_live_events_created ON live_events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_live_events_topic ON live_events(normalized_topic, created_at DESC);

            CREATE TABLE IF NOT EXISTS network_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                editor TEXT NOT NULL,
                article TEXT NOT NULL,
                wiki TEXT NOT NULL,
                wiki_domain TEXT NOT NULL,
                domain TEXT NOT NULL,
                rev_id INTEGER NOT NULL,
                risk_score INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(editor, article, domain, rev_id)
            );

            CREATE INDEX IF NOT EXISTS idx_network_score ON network_edges(risk_score DESC);

            CREATE TABLE IF NOT EXISTS benchmark_cases (
                case_id TEXT PRIMARY KEY,
                wiki_domain TEXT NOT NULL,
                rev_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                user TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                diff_url TEXT NOT NULL,
                ground_truth_label INTEGER NOT NULL,
                category TEXT NOT NULL,
                rationale TEXT NOT NULL,
                heuristic_score INTEGER NOT NULL,
                llm_verdict TEXT NOT NULL,
                llm_confidence REAL NOT NULL,
                techniques_json TEXT NOT NULL DEFAULT '[]',
                triggered_signals_json TEXT NOT NULL DEFAULT '[]',
                guardrails_json TEXT NOT NULL DEFAULT '[]',
                explanation TEXT NOT NULL,
                highlighted_quote TEXT DEFAULT '',
                preset_key TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(wiki_domain, rev_id)
            );
            """
        )
    logger.info("Initialized SQLite database at %s", db_path or settings.db_path)


def save_analyzed_revision(
    revision: dict[str, Any],
    user_profile: dict[str, Any],
    heuristics: dict[str, Any],
    llm_analysis: dict[str, Any],
    source: str = "inspect",
    preset_key: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Upsert a fully analyzed revision and its Editor->Article->Domain network edges."""
    now_iso = _utc_now_iso()
    norm_topic = normalize_topic(revision.get("title", ""))

    # Strip heavy raw HTML from cached detail payload while keeping line-level diffs
    detail_compact = {
        k: v
        for k, v in revision.items()
        if k != "diff_html"
    }

    with get_connection(db_path) as conn:
        if preset_key:
            conn.execute(
                "UPDATE revisions SET preset_key = NULL WHERE preset_key = ? AND NOT (wiki_domain = ? AND rev_id = ?)",
                (preset_key, revision["wiki_domain"], int(revision["rev_id"])),
            )
        conn.execute(
            """
            INSERT INTO revisions (
                wiki_domain, rev_id, parent_id, wiki, title, normalized_topic,
                user, timestamp, comment, diff_url, risk_score, risk_tier,
                added_chars, deleted_chars, added_urls_json, added_domains_json,
                triggered_signals_json, guardrails_json, user_profile_json,
                revision_detail_json, llm_analysis_json, source, preset_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wiki_domain, rev_id) DO UPDATE SET
                risk_score = excluded.risk_score,
                risk_tier = excluded.risk_tier,
                triggered_signals_json = excluded.triggered_signals_json,
                guardrails_json = excluded.guardrails_json,
                user_profile_json = excluded.user_profile_json,
                revision_detail_json = excluded.revision_detail_json,
                llm_analysis_json = excluded.llm_analysis_json,
                source = COALESCE(revisions.source, excluded.source),
                preset_key = COALESCE(excluded.preset_key, revisions.preset_key),
                created_at = excluded.created_at
            """,
            (
                revision["wiki_domain"],
                int(revision["rev_id"]),
                int(revision.get("parent_id") or 0),
                revision.get("wiki", "enwiki"),
                revision.get("title", ""),
                norm_topic,
                revision.get("user", "Unknown"),
                revision.get("timestamp", now_iso),
                revision.get("comment", ""),
                revision.get("diff_url", ""),
                int(heuristics["score"]),
                heuristics["risk_tier"],
                int(revision.get("added_chars") or 0),
                int(revision.get("deleted_chars") or 0),
                json.dumps(revision.get("added_urls", [])),
                json.dumps(heuristics.get("added_domains", [])),
                json.dumps(heuristics.get("triggered_signals", [])),
                json.dumps(heuristics.get("guardrails_applied", [])),
                json.dumps(user_profile),
                json.dumps(detail_compact),
                json.dumps(llm_analysis),
                source,
                preset_key,
                now_iso,
            ),
        )

        # Record Editor -> Article -> External Domain edges (or '(no-external-domain)' marker)
        domains = heuristics.get("added_domains") or []
        for dom in domains:
            conn.execute(
                """
                INSERT OR IGNORE INTO network_edges (
                    editor, article, wiki, wiki_domain, domain, rev_id, risk_score, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.get("user", "Unknown"),
                    revision.get("title", ""),
                    revision.get("wiki", "enwiki"),
                    revision["wiki_domain"],
                    dom,
                    int(revision["rev_id"]),
                    int(heuristics["score"]),
                    revision.get("timestamp", now_iso),
                    now_iso,
                ),
            )

    return build_forensic_dossier(revision=detail_compact, user_profile=user_profile, heuristics=heuristics, llm_analysis=llm_analysis, preset_key=preset_key)


def build_forensic_dossier(
    revision: dict[str, Any],
    user_profile: dict[str, Any],
    heuristics: dict[str, Any],
    llm_analysis: dict[str, Any],
    preset_key: str | None = None,
) -> dict[str, Any]:
    """Assemble standardized forensic dossier response."""
    return {
        "wiki_domain": revision.get("wiki_domain"),
        "wiki": revision.get("wiki"),
        "rev_id": revision.get("rev_id"),
        "parent_id": revision.get("parent_id"),
        "title": revision.get("title"),
        "user": revision.get("user"),
        "timestamp": revision.get("timestamp"),
        "comment": revision.get("comment"),
        "diff_url": revision.get("diff_url"),
        "tags": revision.get("tags", []),
        "is_reverted": revision.get("is_reverted", False),
        "diff": {
            "added_lines": revision.get("added_lines", []),
            "deleted_lines": revision.get("deleted_lines", []),
            "added_chars": revision.get("added_chars", 0),
            "deleted_chars": revision.get("deleted_chars", 0),
            "added_urls": revision.get("added_urls", []),
            "added_domains": revision.get("added_domains", []),
        },
        "user_profile": user_profile,
        "heuristics": heuristics,
        "llm_analysis": llm_analysis,
        "preset_key": preset_key,
    }


def _row_to_dossier(row: sqlite3.Row) -> dict[str, Any]:
    revision = json.loads(row["revision_detail_json"])
    user_profile = json.loads(row["user_profile_json"])
    llm_analysis = json.loads(row["llm_analysis_json"])
    added_domains = json.loads(row["added_domains_json"])
    heuristics = {
        "score": row["risk_score"],
        "risk_tier": row["risk_tier"],
        "triggered_signals": json.loads(row["triggered_signals_json"]),
        "guardrails_applied": json.loads(row["guardrails_json"]),
        "added_domains": added_domains,
    }
    return build_forensic_dossier(
        revision=revision,
        user_profile=user_profile,
        heuristics=heuristics,
        llm_analysis=llm_analysis,
        preset_key=row["preset_key"],
    )


def get_cached_revision(
    wiki_domain: str,
    rev_id: int,
    db_path: str | None = None,
) -> dict[str, Any] | None:
    """Return cached forensic dossier for (wiki_domain, rev_id) if present."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM revisions WHERE wiki_domain = ? AND rev_id = ?",
            (wiki_domain, int(rev_id)),
        ).fetchone()
        if not row:
            return None
        return _row_to_dossier(row)


def save_live_event(
    revision: dict[str, Any],
    heuristics: dict[str, Any],
    llm_analysis: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> None:
    """Persist a scored live Wikimedia SSE event and any domain edges."""
    now_iso = _utc_now_iso()
    norm_topic = normalize_topic(revision.get("title", ""))
    verdict = llm_analysis.get("verdict") if llm_analysis else None
    confidence = llm_analysis.get("confidence") if llm_analysis else None
    explanation = llm_analysis.get("explanation") if llm_analysis else None

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO live_events (
                wiki, wiki_domain, rev_id, parent_id, title, normalized_topic,
                user, timestamp, comment, diff_url, risk_score, risk_tier,
                added_domains_json, triggered_signals_json, guardrails_json,
                verdict, confidence, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wiki_domain, rev_id) DO UPDATE SET
                risk_score = excluded.risk_score,
                risk_tier = excluded.risk_tier,
                verdict = COALESCE(excluded.verdict, live_events.verdict),
                confidence = COALESCE(excluded.confidence, live_events.confidence),
                explanation = COALESCE(excluded.explanation, live_events.explanation)
            """,
            (
                revision.get("wiki", "enwiki"),
                revision["wiki_domain"],
                int(revision["rev_id"]),
                int(revision.get("parent_id") or 0),
                revision.get("title", ""),
                norm_topic,
                revision.get("user", "Unknown"),
                revision.get("timestamp", now_iso),
                revision.get("comment", ""),
                revision.get("diff_url", ""),
                int(heuristics["score"]),
                heuristics["risk_tier"],
                json.dumps(heuristics.get("added_domains", [])),
                json.dumps(heuristics.get("triggered_signals", [])),
                json.dumps(heuristics.get("guardrails_applied", [])),
                verdict,
                confidence,
                explanation,
                now_iso,
            ),
        )

        for dom in heuristics.get("added_domains", []):
            conn.execute(
                """
                INSERT OR IGNORE INTO network_edges (
                    editor, article, wiki, wiki_domain, domain, rev_id, risk_score, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.get("user", "Unknown"),
                    revision.get("title", ""),
                    revision.get("wiki", "enwiki"),
                    revision["wiki_domain"],
                    dom,
                    int(revision["rev_id"]),
                    int(heuristics["score"]),
                    revision.get("timestamp", now_iso),
                    now_iso,
                ),
            )

        # Keep live_events bounded to latest 1000 rows
        conn.execute(
            """
            DELETE FROM live_events
            WHERE id NOT IN (
                SELECT id FROM live_events ORDER BY id DESC LIMIT 1000
            )
            """
        )


def find_cross_wiki_bursts(
    title: str,
    added_domains: list[str],
    current_wiki: str,
    window_minutes: int | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Find recent edits in other language wikis matching the same domain or topic."""
    win = window_minutes or settings.cross_wiki_window_minutes
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=win)).isoformat()
    norm_topic = normalize_topic(title)

    matches: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    with get_connection(db_path) as conn:
        if norm_topic and len(norm_topic) >= 4:
            rows = conn.execute(
                """
                SELECT wiki, wiki_domain, rev_id, title, user, timestamp, added_domains_json
                FROM live_events
                WHERE normalized_topic = ? AND wiki != ? AND created_at >= ?
                ORDER BY id DESC LIMIT 20
                """,
                (norm_topic, current_wiki, cutoff_iso),
            ).fetchall()
            for r in rows:
                key = (r["wiki_domain"], r["rev_id"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    matches.append(dict(r))

        if added_domains:
            rows = conn.execute(
                """
                SELECT wiki, wiki_domain, rev_id, title, user, timestamp, added_domains_json
                FROM live_events
                WHERE wiki != ? AND created_at >= ? AND added_domains_json != '[]'
                ORDER BY id DESC LIMIT 100
                """,
                (current_wiki, cutoff_iso),
            ).fetchall()
            target_set = set(added_domains)
            for r in rows:
                row_doms = set(json.loads(r["added_domains_json"]))
                if target_set & row_doms:
                    key = (r["wiki_domain"], r["rev_id"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        matches.append(dict(r))

    return matches


def get_live_feed(
    limit: int = 50,
    wiki: str | None = None,
    min_score: int = 0,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return latest scored live edits from SQLite."""
    clauses = ["risk_score >= ?"]
    params: list[Any] = [int(min_score)]
    if wiki:
        clauses.append("wiki = ?")
        params.append(wiki)

    where_sql = " AND ".join(clauses)
    params.append(max(1, min(limit, 200)))

    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM live_events
            WHERE {where_sql}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "wiki": r["wiki"],
                "wiki_domain": r["wiki_domain"],
                "rev_id": r["rev_id"],
                "parent_id": r["parent_id"],
                "title": r["title"],
                "user": r["user"],
                "timestamp": r["timestamp"],
                "comment": r["comment"],
                "diff_url": r["diff_url"],
                "risk_score": r["risk_score"],
                "risk_tier": r["risk_tier"],
                "added_domains": json.loads(r["added_domains_json"]),
                "triggered_signals": json.loads(r["triggered_signals_json"]),
                "guardrails_applied": json.loads(r["guardrails_json"]),
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "explanation": r["explanation"],
            }
        )
    return results


def get_presets(db_path: str | None = None) -> list[dict[str, Any]]:
    """Return the 3 cached real preset cases (true_positive, hard_negative, uncertain)."""
    desired_order = ["true_positive", "hard_negative", "uncertain"]
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM revisions
            WHERE preset_key IN ('true_positive', 'hard_negative', 'uncertain')
            ORDER BY created_at DESC
            """
        ).fetchall()

    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        pk = row["preset_key"]
        if pk and pk not in by_key:
            by_key[pk] = _row_to_dossier(row)

    return [by_key[k] for k in desired_order if k in by_key]


def get_network_graph(
    min_score: int = 0,
    limit: int = 120,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Build Editor -> Article -> External Domain graph from SQLite edges."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT editor, article, wiki, wiki_domain, domain, rev_id, risk_score, timestamp
            FROM network_edges
            WHERE risk_score >= ?
            ORDER BY risk_score DESC, id DESC
            LIMIT ?
            """,
            (int(min_score), max(1, min(limit, 500))),
        ).fetchall()

    nodes_map: dict[str, dict[str, Any]] = {}
    edges_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    for r in rows:
        editor_id = f"editor:{r['editor']}"
        article_id = f"article:{r['wiki']}:{r['article']}"
        domain_id = f"domain:{r['domain']}"
        score = int(r["risk_score"])

        if editor_id not in nodes_map:
            nodes_map[editor_id] = {
                "id": editor_id,
                "label": r["editor"],
                "type": "editor",
                "max_risk_score": score,
                "wiki": r["wiki"],
            }
        else:
            nodes_map[editor_id]["max_risk_score"] = max(nodes_map[editor_id]["max_risk_score"], score)

        if article_id not in nodes_map:
            nodes_map[article_id] = {
                "id": article_id,
                "label": r["article"],
                "type": "article",
                "max_risk_score": score,
                "wiki": r["wiki"],
                "wiki_domain": r["wiki_domain"],
            }
        else:
            nodes_map[article_id]["max_risk_score"] = max(nodes_map[article_id]["max_risk_score"], score)

        if domain_id not in nodes_map:
            nodes_map[domain_id] = {
                "id": domain_id,
                "label": r["domain"],
                "type": "domain",
                "max_risk_score": score,
            }
        else:
            nodes_map[domain_id]["max_risk_score"] = max(nodes_map[domain_id]["max_risk_score"], score)

        e1_key = (editor_id, article_id, "edited")
        if e1_key not in edges_map:
            edges_map[e1_key] = {
                "source": editor_id,
                "target": article_id,
                "relation": "edited",
                "weight": 1,
                "max_risk_score": score,
                "rev_ids": [r["rev_id"]],
            }
        else:
            edges_map[e1_key]["weight"] += 1
            edges_map[e1_key]["max_risk_score"] = max(edges_map[e1_key]["max_risk_score"], score)
            if r["rev_id"] not in edges_map[e1_key]["rev_ids"]:
                edges_map[e1_key]["rev_ids"].append(r["rev_id"])

        e2_key = (article_id, domain_id, "links_to")
        if e2_key not in edges_map:
            edges_map[e2_key] = {
                "source": article_id,
                "target": domain_id,
                "relation": "links_to",
                "weight": 1,
                "max_risk_score": score,
                "rev_ids": [r["rev_id"]],
            }
        else:
            edges_map[e2_key]["weight"] += 1
            edges_map[e2_key]["max_risk_score"] = max(edges_map[e2_key]["max_risk_score"], score)
            if r["rev_id"] not in edges_map[e2_key]["rev_ids"]:
                edges_map[e2_key]["rev_ids"].append(r["rev_id"])

    nodes = list(nodes_map.values())
    edges = list(edges_map.values())
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "editors": sum(1 for n in nodes if n["type"] == "editor"),
            "articles": sum(1 for n in nodes if n["type"] == "article"),
            "domains": sum(1 for n in nodes if n["type"] == "domain"),
        },
    }


def save_benchmark_case(case: dict[str, Any], db_path: str | None = None) -> None:
    """Persist a single real benchmark evaluation case to SQLite."""
    now_iso = _utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO benchmark_cases (
                case_id, wiki_domain, rev_id, title, user, timestamp, diff_url,
                ground_truth_label, category, rationale, heuristic_score,
                llm_verdict, llm_confidence, techniques_json, triggered_signals_json,
                guardrails_json, explanation, highlighted_quote, preset_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wiki_domain, rev_id) DO UPDATE SET
                case_id = excluded.case_id,
                ground_truth_label = excluded.ground_truth_label,
                category = excluded.category,
                rationale = excluded.rationale,
                heuristic_score = excluded.heuristic_score,
                llm_verdict = excluded.llm_verdict,
                llm_confidence = excluded.llm_confidence,
                techniques_json = excluded.techniques_json,
                triggered_signals_json = excluded.triggered_signals_json,
                guardrails_json = excluded.guardrails_json,
                explanation = excluded.explanation,
                highlighted_quote = excluded.highlighted_quote,
                preset_key = excluded.preset_key,
                created_at = excluded.created_at
            """,
            (
                case["case_id"],
                case["wiki_domain"],
                int(case["rev_id"]),
                case["title"],
                case["user"],
                case["timestamp"],
                case["diff_url"],
                int(case["ground_truth_label"]),
                case["category"],
                case["rationale"],
                int(case["heuristic_score"]),
                case["llm_verdict"],
                float(case["llm_confidence"]),
                json.dumps(case.get("techniques", [])),
                json.dumps(case.get("triggered_signals", [])),
                json.dumps(case.get("guardrails_applied", [])),
                case.get("explanation", ""),
                case.get("highlighted_quote", ""),
                case.get("preset_key"),
                now_iso,
            ),
        )


def compute_metrics_at_threshold(cases: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    """Compute confusion matrix, Precision, Recall, F1, Accuracy, and FPR at threshold."""
    tp = fp = tn = fn = 0
    evaluated_cases: list[dict[str, Any]] = []

    for c in cases:
        truth = int(c["ground_truth_label"]) == 1
        score = int(c["heuristic_score"])
        predicted_positive = score >= threshold

        if truth and predicted_positive:
            tp += 1
            outcome = "TP"
        elif not truth and predicted_positive:
            fp += 1
            outcome = "FP"
        elif not truth and not predicted_positive:
            tn += 1
            outcome = "TN"
        else:
            fn += 1
            outcome = "FN"

        evaluated_cases.append({**c, "predicted_positive": predicted_positive, "outcome": outcome})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(cases) if cases else 0.0

    return {
        "threshold": threshold,
        "total_cases": len(cases),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "accuracy": round(accuracy, 4),
        },
        "cases": evaluated_cases,
    }


def get_benchmark_report(
    threshold: int | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Load stored benchmark cases from SQLite and compute metrics at `threshold`."""
    active_threshold = settings.risk_threshold_default if threshold is None else max(0, min(100, int(threshold)))
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM benchmark_cases ORDER BY ground_truth_label DESC, heuristic_score DESC").fetchall()

    cases: list[dict[str, Any]] = []
    for r in rows:
        cases.append(
            {
                "case_id": r["case_id"],
                "wiki_domain": r["wiki_domain"],
                "rev_id": r["rev_id"],
                "title": r["title"],
                "user": r["user"],
                "timestamp": r["timestamp"],
                "diff_url": r["diff_url"],
                "ground_truth_label": r["ground_truth_label"],
                "category": r["category"],
                "rationale": r["rationale"],
                "heuristic_score": r["heuristic_score"],
                "llm_verdict": r["llm_verdict"],
                "llm_confidence": r["llm_confidence"],
                "techniques": json.loads(r["techniques_json"]),
                "triggered_signals": json.loads(r["triggered_signals_json"]),
                "guardrails_applied": json.loads(r["guardrails_json"]),
                "explanation": r["explanation"],
                "highlighted_quote": r["highlighted_quote"],
                "preset_key": r["preset_key"],
            }
        )

    report = compute_metrics_at_threshold(cases, active_threshold)
    threshold_sweep = [
        {
            k: v
            for k, v in compute_metrics_at_threshold(cases, t).items()
            if k in {"threshold", "confusion_matrix", "metrics"}
        }
        for t in (30, 40, 50, 60, 70)
    ]
    report["threshold_sweep"] = threshold_sweep
    return report


def get_db_stats(db_path: str | None = None) -> dict[str, int]:
    """Return row counts for healthcheck telemetry."""
    with get_connection(db_path) as conn:
        rev_count = conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
        live_count = conn.execute("SELECT COUNT(*) FROM live_events").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM network_edges").fetchone()[0]
        bench_count = conn.execute("SELECT COUNT(*) FROM benchmark_cases").fetchone()[0]
    return {
        "cached_revisions": int(rev_count),
        "live_events": int(live_count),
        "network_edges": int(edge_count),
        "benchmark_cases": int(bench_count),
    }
