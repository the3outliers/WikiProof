"""Real-data benchmark seeder and threshold evaluator for WikiProof.

Fetches genuine historical Wikipedia revisions via the MediaWiki Action API—
combining documented reverted/COI/whitewashing/link-injection revisions with
legitimate volunteer edits on Featured Articles—runs each through heuristics.py
and llm_analyzer.py, persists them into SQLite + data/benchmark_results.json,
and prints Precision, Recall, F1, False-Positive Rate, and Confusion Matrix.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import get_logger, settings
from database import (
    get_benchmark_report,
    init_db,
    save_analyzed_revision,
    save_benchmark_case,
)
from heuristics import score_revision
from llm_analyzer import analyze_revision
from wikimedia_client import fetch_revision_details, fetch_user_profile

logger = get_logger("wikiproof.seed_benchmark")


@dataclass(frozen=True)
class BenchmarkSeedSpec:
    """Definition of a real historical Wikipedia revision for benchmark evaluation."""

    case_id: str
    wiki_domain: str
    rev_id: int
    ground_truth_label: int  # 1 = Manipulative/Reverted/Bad-faith, 0 = Legitimate volunteer edit
    category: str
    rationale: str
    preset_key: str | None = None


BENCHMARK_SPECS: list[BenchmarkSeedSpec] = [
    # --- Positive Cases (Documented Reverted COI / Whitewashing / Link Injection) ---
    BenchmarkSeedSpec(
        case_id="tp_tamika_upton_coi",
        wiki_domain="en.wikipedia.org",
        rev_id=1385610575,
        ground_truth_label=1,
        category="reverted_coi_whitewashing",
        rationale="Reverted self-editing COI revision by subject-named account removing 1.9KB of sourced text.",
    ),
    BenchmarkSeedSpec(
        case_id="tp_jahangirabad_coi",
        wiki_domain="en.wikipedia.org",
        rev_id=1385617303,
        ground_truth_label=1,
        category="reverted_coi_whitewashing",
        rationale="Reverted mass deletion (-15KB) by brand-new account matching article name.",
    ),
    BenchmarkSeedSpec(
        case_id="tp_ilam_university_links",
        wiki_domain="en.wikipedia.org",
        rev_id=1385614788,
        ground_truth_label=1,
        category="reverted_link_injection",
        rationale="Reverted institutional self-promotion injecting unverified external subdomains.",
        preset_key="true_positive",
    ),
    BenchmarkSeedSpec(
        case_id="tp_kansas_health_reddit",
        wiki_domain="en.wikipedia.org",
        rev_id=1385606898,
        ground_truth_label=1,
        category="reverted_source_laundering",
        rationale="Reverted edit by new account injecting reddit.com link as 'first hand acct'.",
    ),
    BenchmarkSeedSpec(
        case_id="tp_forrest_gump_blanking",
        wiki_domain="en.wikipedia.org",
        rev_id=1385609523,
        ground_truth_label=1,
        category="reverted_content_deletion",
        rationale="Reverted unexplained section removal (-1.2KB) by low-activity account.",
    ),
    BenchmarkSeedSpec(
        case_id="tp_dodge_concept_links",
        wiki_domain="en.wikipedia.org",
        rev_id=1385609680,
        ground_truth_label=1,
        category="reverted_link_injection",
        rationale="Reverted temporary-account edit injecting unverified external image/blog URLs.",
    ),
    BenchmarkSeedSpec(
        case_id="tp_bodrum_historical_erasure",
        wiki_domain="en.wikipedia.org",
        rev_id=1385616046,
        ground_truth_label=1,
        category="reverted_pov_erasure",
        rationale="Reverted temporary-account edit deleting historical name context on Bodrum.",
        preset_key="uncertain",
    ),
    # --- Negative Cases (Legitimate Volunteer Edits on Featured / High-Visibility Articles) ---
    BenchmarkSeedSpec(
        case_id="tn_climate_change_hard_neg",
        wiki_domain="en.wikipedia.org",
        rev_id=1383408659,
        ground_truth_label=0,
        category="featured_hard_negative",
        rationale="Veteran editor (>70k edits) removing inaccurate phrasing on Climate change article.",
    ),
    BenchmarkSeedSpec(
        case_id="tn_reuters_citation",
        wiki_domain="en.wikipedia.org",
        rev_id=1351299528,
        ground_truth_label=0,
        category="featured_verified_citation",
        rationale="Legitimate volunteer edit adding a Reuters citation alongside industry reporting on Reuters article.",
        preset_key="hard_negative",
    ),
    BenchmarkSeedSpec(
        case_id="tn_fabelmans_archive_cite",
        wiki_domain="en.wikipedia.org",
        rev_id=1385619472,
        ground_truth_label=0,
        category="featured_verified_citation",
        rationale="Veteran editor (>10k edits) rescuing reference via web.archive.org.",
    ),
    BenchmarkSeedSpec(
        case_id="tn_turing_machine_dab",
        wiki_domain="en.wikipedia.org",
        rev_id=1385514535,
        ground_truth_label=0,
        category="featured_maintenance",
        rationale="Veteran editor disambiguating wikilink on Turing machine Featured Article.",
    ),
    BenchmarkSeedSpec(
        case_id="tn_apollo11_copyedit",
        wiki_domain="en.wikipedia.org",
        rev_id=1383944894,
        ground_truth_label=0,
        category="featured_copyedit",
        rationale="Veteran editor copyediting terminology on Apollo 11 Featured Article.",
    ),
    BenchmarkSeedSpec(
        case_id="tn_apollo11_caption",
        wiki_domain="en.wikipedia.org",
        rev_id=1381298063,
        ground_truth_label=0,
        category="featured_copyedit",
        rationale="Veteran editor (>300k edits) adding wikilink to photo caption on Apollo 11.",
    ),
    BenchmarkSeedSpec(
        case_id="tn_photosynthesis_cleanup",
        wiki_domain="en.wikipedia.org",
        rev_id=1370173331,
        ground_truth_label=0,
        category="featured_maintenance",
        rationale="Veteran editor removing redundant 'See also' link already linked in Photosynthesis.",
    ),
    BenchmarkSeedSpec(
        case_id="tn_finland_deprecated_source",
        wiki_domain="en.wikipedia.org",
        rev_id=1384419846,
        ground_truth_label=0,
        category="featured_hard_negative",
        rationale="Established editor removing deprecated source on Finland Featured Article.",
    ),
]


async def seed_all_benchmarks(threshold: int | None = None) -> dict[str, Any]:
    """Fetch real revisions, score them, run LLM analysis, and save benchmark outputs."""
    init_db()
    logger.info("Starting WikiProof real benchmark seeding (%d cases)", len(BENCHMARK_SPECS))

    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
        for spec in BENCHMARK_SPECS:
            logger.info("Processing benchmark case=%s (%s rev=%d)", spec.case_id, spec.wiki_domain, spec.rev_id)
            revision = await fetch_revision_details(spec.wiki_domain, spec.rev_id, client=client)
            user_profile = await fetch_user_profile(spec.wiki_domain, revision["user"], client=client)
            heuristics = score_revision(revision, user_profile, cross_wiki_matches=None)
            llm_result = await analyze_revision(revision, user_profile, heuristics)

            save_analyzed_revision(
                revision=revision,
                user_profile=user_profile,
                heuristics=heuristics,
                llm_analysis=llm_result,
                source="benchmark",
                preset_key=spec.preset_key,
            )

            save_benchmark_case(
                {
                    "case_id": spec.case_id,
                    "wiki_domain": spec.wiki_domain,
                    "rev_id": spec.rev_id,
                    "title": revision["title"],
                    "user": revision["user"],
                    "timestamp": revision["timestamp"],
                    "diff_url": revision["diff_url"],
                    "ground_truth_label": spec.ground_truth_label,
                    "category": spec.category,
                    "rationale": spec.rationale,
                    "heuristic_score": heuristics["score"],
                    "llm_verdict": llm_result["verdict"],
                    "llm_confidence": llm_result["confidence"],
                    "techniques": llm_result["techniques"],
                    "triggered_signals": heuristics["triggered_signals"],
                    "guardrails_applied": heuristics["guardrails_applied"],
                    "explanation": llm_result["explanation"],
                    "highlighted_quote": llm_result["highlighted_quote"],
                    "preset_key": spec.preset_key,
                }
            )

    active_threshold = settings.risk_threshold_default if threshold is None else threshold
    report = get_benchmark_report(threshold=active_threshold)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    out_path = Path(settings.benchmark_results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved benchmark report to %s", out_path)
    return report


def _print_report(report: dict[str, Any]) -> None:
    cm = report["confusion_matrix"]
    m = report["metrics"]
    print("\n=== WikiProof Real-Data Benchmark Report ===")
    print(f"Threshold: {report['threshold']} | Cases: {report['total_cases']}")
    print(
        f"Confusion Matrix -> TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}"
    )
    print(
        f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | "
        f"F1: {m['f1']:.4f} | FPR: {m['false_positive_rate']:.4f} | Accuracy: {m['accuracy']:.4f}"
    )
    print("\n--- Threshold Sweep ---")
    for row in report.get("threshold_sweep", []):
        rcm = row["confusion_matrix"]
        rm = row["metrics"]
        print(
            f"t={row['threshold']:2d} | TP={rcm['tp']:2d} FP={rcm['fp']:2d} TN={rcm['tn']:2d} FN={rcm['fn']:2d} "
            f"| P={rm['precision']:.3f} R={rm['recall']:.3f} F1={rm['f1']:.3f} FPR={rm['false_positive_rate']:.3f}"
        )
    print("\n--- Case Breakdown ---")
    for c in report["cases"]:
        print(
            f"[{c['outcome']:2s}] truth={c['ground_truth_label']} score={c['heuristic_score']:3d} "
            f"llm={c['llm_verdict']:10s} | rev={c['rev_id']} | {c['title'][:28]:28s} | {c['user']}"
        )


if __name__ == "__main__":
    result = asyncio.run(seed_all_benchmarks())
    _print_report(result)
