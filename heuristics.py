"""Deterministic behavioral risk scoring engine with false-positive guardrails.

Scores Wikipedia revisions on a 0-100 scale using transparent behavioral risk
signals (account age/editcount, unverified external link injection, large
content/citation deletion, cross-language burst activity) balanced against
explicit false-positive guardrails (veteran editor history, high-reliability
or academic citations, and routine maintenance/anti-vandalism edit summaries).
"""

from __future__ import annotations

import re
from typing import Any

from config import settings
from wikimedia_client import normalize_domain

HIGH_RELIABILITY_DOMAINS: set[str] = {
    # Wire services & public broadcasters
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "yle.fi",
    "dw.com",
    "pbs.org",
    "npr.org",
    # Academic, scientific & peer-reviewed repositories
    "nature.com",
    "doi.org",
    "dx.doi.org",
    "science.org",
    "sciencemag.org",
    "thelancet.com",
    "nejm.org",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nih.gov",
    "arxiv.org",
    "jstor.org",
    "springer.com",
    "link.springer.com",
    "oup.com",
    "academic.oup.com",
    "cambridge.org",
    "ieee.org",
    "acm.org",
    "plos.org",
    "journals.plos.org",
    "tandfonline.com",
    "wiley.com",
    "sciencedirect.com",
    "scholar.google.com",
    # Official intergovernmental, governmental & encyclopedic sources
    "europa.eu",
    "who.int",
    "un.org",
    "gov.uk",
    "nasa.gov",
    "loc.gov",
    "britannica.com",
    "archive.org",
    "web.archive.org",
    # Established newspapers of record
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "ft.com",
    "economist.com",
    "hs.fi",
    "spiegel.de",
    "lemonde.fr",
}

INTERNAL_WIKI_DOMAINS: set[str] = {
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "wiktionary.org",
    "wikisource.org",
    "wikiquote.org",
    "wikivoyage.org",
    "commons.wikimedia.org",
}

OFFICIAL_DOMAIN_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".gov.uk",
    ".gc.ca",
    ".gov.bc.ca",
    ".edu",
    ".ac.uk",
    ".europa.eu",
)

MAINTENANCE_SUMMARY_REGEX = re.compile(
    r"\b("
    r"typo|typos|copyedit|copy-edit|copy\s*edit|rvv|rv/?v|revert|reverted|reverting|"
    r"undid|undo|vandal|vandalism|grammar|punctuation|spelling|wikify|cleanup|clean\s*up|"
    r"ce|sp|dab|disambig|disambiguation|ref\s*fill|citation\s*bot|iabot|archive\s*link|"
    r"fix\s*link|broken\s*link|dead\s*link|formatting|cat|category|template"
    r")\b",
    re.IGNORECASE,
)

CITATION_MARKER_REGEX = re.compile(
    r"(<ref|</ref>|\{\{\s*cite|https?://|\bdoi\s*=|\bisbn\s*=|\bpmid\s*=)",
    re.IGNORECASE,
)

PROMOTIONAL_PHRASE_REGEX = re.compile(
    r"\b("
    r"leading provider|world-class|renowned|groundbreaking|pioneering|award-winning|"
    r"industry-leading|cutting-edge|visionary|prestigious|best-selling|premier"
    r")\b",
    re.IGNORECASE,
)


def is_domain_in_set(domain: str, domain_set: set[str]) -> bool:
    """Return True if domain matches an entry in domain_set or is a subdomain of one."""
    d = normalize_domain(domain)
    if not d:
        return False
    if d in domain_set:
        return True
    return any(d.endswith(f".{trusted}") for trusted in domain_set)


def is_official_gov_or_academic_domain(domain: str) -> bool:
    """Return True if domain ends with an official government or academic suffix."""
    d = normalize_domain(domain)
    if not d:
        return False
    return any(d == suffix.lstrip(".") or d.endswith(suffix) for suffix in OFFICIAL_DOMAIN_SUFFIXES)


def is_high_reliability_domain(domain: str) -> bool:
    """Return True if domain is an official gov/academic domain or in the curated whitelist."""
    return is_official_gov_or_academic_domain(domain) or is_domain_in_set(domain, HIGH_RELIABILITY_DOMAINS)


def is_internal_wiki_domain(domain: str) -> bool:
    """Return True if domain belongs to Wikimedia infrastructure."""
    return is_domain_in_set(domain, INTERNAL_WIKI_DOMAINS)


def normalize_topic(title: str) -> str:
    """Normalize an article title for cross-language burst matching."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", title or "")
    cleaned = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F\u0400-\u04FF]+", " ", cleaned).strip().lower()
    return cleaned


def score_revision(
    revision: dict[str, Any],
    user_profile: dict[str, Any],
    cross_wiki_matches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute deterministic behavioral risk score (0-100), signals, and guardrails."""
    triggered_signals: list[dict[str, Any]] = []
    guardrails_applied: list[dict[str, Any]] = []
    raw_score = 0

    age_days = float(user_profile.get("age_days") or 0.0)
    editcount = int(user_profile.get("editcount") or 0)
    is_anonymous = bool(user_profile.get("is_anonymous"))
    is_temp = bool(user_profile.get("is_temp"))

    added_chars = int(revision.get("added_chars") or 0)
    deleted_chars = int(revision.get("deleted_chars") or 0)
    added_text = revision.get("added_text") or ""
    deleted_text = revision.get("deleted_text") or ""
    comment = (revision.get("comment") or "").strip()
    title = revision.get("title") or ""
    username = user_profile.get("username") or revision.get("user") or ""

    raw_domains = revision.get("added_domains") or []
    external_domains = [d for d in raw_domains if not is_internal_wiki_domain(d)]
    official_domains = [d for d in external_domains if is_official_gov_or_academic_domain(d)]
    verified_domains = [d for d in external_domains if is_high_reliability_domain(d)]
    unverified_domains = [d for d in external_domains if not is_high_reliability_domain(d)]

    # --- Signal 1: Single-Purpose / New / Temporary Account ---
    if is_anonymous or is_temp:
        weight = 25
        raw_score += weight
        triggered_signals.append(
            {
                "id": "anonymous_or_temp_account",
                "name": "Anonymous or Temporary Account",
                "weight": weight,
                "detail": f"Editor '{username}' is using a temporary or unregistered IP account.",
            }
        )
    elif age_days < 30 or editcount < 50:
        weight = 30 if (age_days < 7 and editcount < 15) else 25
        raw_score += weight
        triggered_signals.append(
            {
                "id": "new_or_single_purpose_account",
                "name": "Single-Purpose / New Account",
                "weight": weight,
                "detail": f"Account age is {age_days:.1f} days with {editcount} total edits.",
            }
        )

    # --- Signal 2: Unverified External Link Injection ---
    if unverified_domains:
        weight = min(35, 25 + 5 * (len(unverified_domains) - 1))
        raw_score += weight
        triggered_signals.append(
            {
                "id": "unverified_external_link",
                "name": "Unverified External Link Injection",
                "weight": weight,
                "detail": f"Added {len(unverified_domains)} non-whitelisted domain(s): {', '.join(unverified_domains[:5])}.",
            }
        )

    # --- Signal 3: Large Content / Citation Deletion ---
    minimal_replacement = added_chars < max(120, int(deleted_chars * 0.35))
    if deleted_chars > 300 and minimal_replacement:
        has_deleted_citations = bool(CITATION_MARKER_REGEX.search(deleted_text))
        weight = 32 if has_deleted_citations else 25
        raw_score += weight
        triggered_signals.append(
            {
                "id": "large_content_or_citation_deletion",
                "name": "Large Content / Citation Deletion",
                "weight": weight,
                "detail": (
                    f"Removed {deleted_chars} chars (replaced with {added_chars} chars)"
                    + (" including citation/reference markup." if has_deleted_citations else ".")
                ),
            }
        )

    # --- Signal 4: Cross-Language Burst Detection ---
    burst_details: dict[str, Any] | None = None
    if cross_wiki_matches:
        distinct_wikis = sorted({m.get("wiki", "") for m in cross_wiki_matches if m.get("wiki")})
        current_wiki = revision.get("wiki") or ""
        if current_wiki and current_wiki not in distinct_wikis:
            distinct_wikis = sorted(set(distinct_wikis) | {current_wiki})
        if len(distinct_wikis) >= 2:
            weight = 25
            raw_score += weight
            burst_details = {
                "wikis": distinct_wikis,
                "match_count": len(cross_wiki_matches),
                "window_minutes": settings.cross_wiki_window_minutes,
            }
            triggered_signals.append(
                {
                    "id": "cross_language_burst",
                    "name": "Cross-Language Burst Detection",
                    "weight": weight,
                    "detail": (
                        f"Same domain or topic edited across {len(distinct_wikis)} language wikis "
                        f"({', '.join(distinct_wikis)}) within {settings.cross_wiki_window_minutes} minutes."
                    ),
                }
            )

    # --- Supporting Behavioral Cues: Promotional Language / COI Username Overlap ---
    promo_matches = PROMOTIONAL_PHRASE_REGEX.findall(added_text)
    if promo_matches and added_chars > 120:
        weight = 15
        raw_score += weight
        triggered_signals.append(
            {
                "id": "promotional_peacock_framing",
                "name": "Promotional / Peacock Phrasing",
                "weight": weight,
                "detail": f"Inserted promotional phrasing ({', '.join(sorted({m.lower() for m in promo_matches})[:4])}).",
            }
        )

    title_tokens = {t for t in normalize_topic(title).split() if len(t) >= 4}
    user_tokens = {t for t in normalize_topic(username).split() if len(t) >= 4}
    if title_tokens and user_tokens and (title_tokens & user_tokens) and editcount < 200:
        weight = 20
        raw_score += weight
        triggered_signals.append(
            {
                "id": "coi_username_article_overlap",
                "name": "Potential COI Username Overlap",
                "weight": weight,
                "detail": f"Username '{username}' overlaps with article title '{title}' ({sorted(title_tokens & user_tokens)}).",
            }
        )

    if added_chars > 250 and not CITATION_MARKER_REGEX.search(added_text) and editcount < 100:
        weight = 12
        raw_score += weight
        triggered_signals.append(
            {
                "id": "unsourced_substantive_addition",
                "name": "Unsourced Substantive Addition",
                "weight": weight,
                "detail": f"Low-edit account added {added_chars} chars without citation markup or external references.",
            }
        )

    # =========================================================================
    # False-Positive Guardrails
    # =========================================================================

    # Guardrail 1: Veteran Editor
    if editcount > 2000 and age_days > 365:
        adjustment = -30 if editcount > 10000 else -25
        raw_score += adjustment
        guardrails_applied.append(
            {
                "id": "veteran_editor",
                "name": "Veteran Editor Guardrail",
                "adjustment": adjustment,
                "detail": f"Editor has {editcount:,} edits across {age_days:.0f} days (~{age_days / 365.25:.1f} years).",
            }
        )

    # Guardrail 2: Official Government / Academic & High-Reliability Citations
    if official_domains:
        adjustment = -25 if not unverified_domains else -15
        raw_score += adjustment
        guardrails_applied.append(
            {
                "id": "official_gov_academic_source",
                "name": "Official Government / Academic Source",
                "adjustment": adjustment,
                "detail": f"Added official government/academic domain(s): {', '.join(official_domains[:5])}.",
            }
        )
    elif verified_domains and not unverified_domains:
        adjustment = -20
        raw_score += adjustment
        guardrails_applied.append(
            {
                "id": "high_reliability_citations",
                "name": "High-Reliability / Academic Citations",
                "adjustment": adjustment,
                "detail": f"All added external links point to vetted sources: {', '.join(verified_domains[:5])}.",
            }
        )

    # Guardrail 3: Routine Maintenance / Anti-Vandalism Summary
    if comment and MAINTENANCE_SUMMARY_REGEX.search(comment) and not unverified_domains:
        adjustment = -20
        raw_score += adjustment
        guardrails_applied.append(
            {
                "id": "routine_maintenance_summary",
                "name": "Routine Maintenance / Anti-Vandalism Summary",
                "adjustment": adjustment,
                "detail": f"Edit summary indicates routine maintenance or counter-vandalism: '{comment[:100]}'.",
            }
        )

    # Guardrail 4: Minor Copyedit / Micro-Edit
    if added_chars <= 40 and deleted_chars <= 40 and not external_domains:
        adjustment = -10
        raw_score += adjustment
        guardrails_applied.append(
            {
                "id": "minor_micro_edit",
                "name": "Minor Text / Formatting Adjustment",
                "adjustment": adjustment,
                "detail": f"Small change (+{added_chars}/-{deleted_chars} chars) with no external link additions.",
            }
        )

    final_score = max(0, min(100, int(round(raw_score))))
    if final_score >= settings.risk_threshold_suspicious:
        risk_tier = "high"
    elif final_score >= settings.risk_threshold_uncertain:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    return {
        "score": final_score,
        "raw_score": raw_score,
        "risk_tier": risk_tier,
        "triggered_signals": triggered_signals,
        "guardrails_applied": guardrails_applied,
        "added_domains": external_domains,
        "verified_domains": verified_domains,
        "unverified_domains": unverified_domains,
        "cross_wiki_burst": burst_details,
    }
