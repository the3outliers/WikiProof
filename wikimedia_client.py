"""Wikimedia EventStreams SSE consumer and MediaWiki Action API client.

Connects to Wikimedia's live recentchange SSE stream (filtered to edits in
article namespace 0 by non-bots across configured Wikipedia language editions),
parses revision URLs, fetches exact revision diffs via MediaWiki compare API,
extracts added/deleted lines and newly injected external URLs/domains, and
retrieves editor account age, edit count, and groups.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from config import get_logger, settings

logger = get_logger("wikiproof.wikimedia_client")

WIKI_TO_DOMAIN: dict[str, str] = {
    "enwiki": "en.wikipedia.org",
    "fiwiki": "fi.wikipedia.org",
    "dewiki": "de.wikipedia.org",
    "ruwiki": "ru.wikipedia.org",
    "svwiki": "sv.wikipedia.org",
}

DOMAIN_TO_WIKI: dict[str, str] = {v: k for k, v in WIKI_TO_DOMAIN.items()}

URL_REGEX = re.compile(r"https?://[^\s\]\|\"'<>{}\)]+", re.IGNORECASE)
IPV4_REGEX = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
IPV6_REGEX = re.compile(r"^[0-9a-fA-F:]+$")


def wiki_to_domain(wiki: str) -> str:
    """Convert a Wikimedia DB name (e.g., enwiki) to its Wikipedia domain."""
    if wiki in WIKI_TO_DOMAIN:
        return WIKI_TO_DOMAIN[wiki]
    if wiki.endswith("wiki") and len(wiki) > 4:
        return f"{wiki[:-4]}.wikipedia.org"
    return "en.wikipedia.org"


def domain_to_wiki(wiki_domain: str) -> str:
    """Convert a Wikipedia domain (e.g., en.wikipedia.org) to its wiki DB name."""
    normalized = wiki_domain.lower().strip()
    if normalized in DOMAIN_TO_WIKI:
        return DOMAIN_TO_WIKI[normalized]
    if normalized.endswith(".wikipedia.org"):
        lang = normalized.split(".")[0]
        return f"{lang}wiki"
    return "enwiki"


def normalize_domain(url_or_domain: str) -> str:
    """Extract and normalize a lowercase hostname without leading www."""
    candidate = url_or_domain.strip()
    if "://" in candidate:
        host = urlparse(candidate).netloc.lower()
    else:
        host = candidate.split("/")[0].lower()
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_urls_and_domains(text: str) -> tuple[list[str], list[str]]:
    """Extract unique external URLs and normalized domains from diff text."""
    if not text:
        return [], []
    urls: list[str] = []
    seen_urls: set[str] = set()
    domains: list[str] = []
    seen_domains: set[str] = set()

    for match in URL_REGEX.findall(text):
        cleaned = match.rstrip(".,;:!?")
        if cleaned not in seen_urls:
            seen_urls.add(cleaned)
            urls.append(cleaned)
        dom = normalize_domain(cleaned)
        if dom and dom not in seen_domains:
            seen_domains.add(dom)
            domains.append(dom)

    return urls, domains


def parse_revision_input(
    url: str | None = None,
    wiki_domain: str | None = None,
    rev_id: int | None = None,
) -> tuple[str, int]:
    """Resolve a Wikipedia revision URL or (wiki_domain, rev_id) pair.

    Supports formats such as:
    - https://en.wikipedia.org/w/index.php?title=Foo&diff=123456&oldid=123450
    - https://en.wikipedia.org/w/index.php?diff=prev&oldid=123456
    - https://fi.wikipedia.org/wiki/Special:Diff/123456
    - https://de.wikipedia.org/wiki/Special:PermanentLink/123456
    """
    if url:
        parsed = urlparse(url.strip())
        if not parsed.netloc:
            raise ValueError(f"Invalid Wikipedia revision URL: {url}")
        resolved_domain = parsed.netloc.lower()
        if resolved_domain.startswith("m."):
            resolved_domain = resolved_domain[2:]
        if ".m.wikipedia.org" in resolved_domain:
            resolved_domain = resolved_domain.replace(".m.wikipedia.org", ".wikipedia.org")

        qs = parse_qs(parsed.query)
        diff_vals = qs.get("diff", [])
        oldid_vals = qs.get("oldid", [])

        if diff_vals and diff_vals[0].isdigit():
            return resolved_domain, int(diff_vals[0])

        # Special:Diff/<rev> or Special:Diff/<from>/<to> or Special:PermanentLink/<rev>
        path_parts = [p for p in parsed.path.split("/") if p]
        for idx, part in enumerate(path_parts):
            if part.startswith("Special:Diff") or part.startswith("Special:PermanentLink"):
                numeric_tail = [int(p) for p in path_parts[idx + 1 :] if p.isdigit()]
                if numeric_tail:
                    return resolved_domain, numeric_tail[-1]

        if oldid_vals and oldid_vals[0].isdigit():
            return resolved_domain, int(oldid_vals[0])

        raise ValueError(f"Could not extract revision ID from URL: {url}")

    if wiki_domain and rev_id:
        clean_domain = wiki_domain.strip().lower()
        if "://" in clean_domain:
            clean_domain = urlparse(clean_domain).netloc.lower()
        return clean_domain, int(rev_id)

    raise ValueError("Provide either a Wikipedia revision `url` or both `wiki_domain` and `rev_id`.")


def _parse_compare_html(diff_html: str) -> dict[str, Any]:
    """Parse MediaWiki action=compare diff HTML for added and deleted content."""
    if not diff_html:
        return {
            "added_lines": [],
            "deleted_lines": [],
            "added_highlights": [],
            "deleted_highlights": [],
            "added_text": "",
            "deleted_text": "",
            "added_chars": 0,
            "deleted_chars": 0,
        }

    soup = BeautifulSoup(f"<table>{diff_html}</table>", "html.parser")
    added_lines: list[str] = []
    deleted_lines: list[str] = []
    added_highlights: list[str] = []
    deleted_highlights: list[str] = []

    for cell in soup.select("td.diff-addedline"):
        line_text = cell.get_text(" ", strip=True)
        if line_text:
            added_lines.append(line_text)
        ins_tags = cell.select("ins.diffchange")
        if ins_tags:
            for ins in ins_tags:
                seg = ins.get_text(" ", strip=True)
                if seg:
                    added_highlights.append(seg)
        elif line_text:
            added_highlights.append(line_text)

    for cell in soup.select("td.diff-deletedline"):
        line_text = cell.get_text(" ", strip=True)
        if line_text:
            deleted_lines.append(line_text)
        del_tags = cell.select("del.diffchange")
        if del_tags:
            for d in del_tags:
                seg = d.get_text(" ", strip=True)
                if seg:
                    deleted_highlights.append(seg)
        elif line_text:
            deleted_highlights.append(line_text)

    added_text = "\n".join(added_lines)
    deleted_text = "\n".join(deleted_lines)

    return {
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "added_highlights": added_highlights,
        "deleted_highlights": deleted_highlights,
        "added_text": added_text,
        "deleted_text": deleted_text,
        "added_chars": len(added_text),
        "deleted_chars": len(deleted_text),
    }


_API_MAX_RETRIES = 3
_api_cooldown_until = 0.0


async def _api_get(
    client: httpx.AsyncClient,
    wiki_domain: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Execute a logged GET request to the MediaWiki Action API."""
    global _api_cooldown_until
    endpoint = f"https://{wiki_domain}/w/api.php"
    full_params = {**params, "format": "json"}
    loop = asyncio.get_running_loop()
    for attempt in range(_API_MAX_RETRIES + 1):
        # Honor any shared cooldown set after a 429 so all workers back off together.
        wait = _api_cooldown_until - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        logger.info("MediaWiki API GET %s params=%s", endpoint, full_params)
        response = await client.get(
            endpoint,
            params=full_params,
            headers={"User-Agent": settings.wiki_user_agent},
        )
        if response.status_code not in (429, 503) or attempt == _API_MAX_RETRIES:
            break
        try:
            retry_after = float(response.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = 2.0 * (2**attempt)
        retry_after = min(max(retry_after, 1.0), 30.0)
        _api_cooldown_until = max(_api_cooldown_until, loop.time() + retry_after)
        logger.warning(
            "MediaWiki API %s on %s; backing off %.1fs (attempt %d)",
            response.status_code, wiki_domain, retry_after, attempt + 1,
        )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"MediaWiki API error ({err.get('code')}): {err.get('info')}")
    return data


async def fetch_revision_details(
    wiki_domain: str,
    rev_id: int,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch revision metadata, parentid, exact +/- diff lines, and added URLs."""
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)

    try:
        assert client is not None
        rev_query = await _api_get(
            client,
            wiki_domain,
            {
                "action": "query",
                "prop": "revisions",
                "revids": str(rev_id),
                "rvprop": "ids|timestamp|user|comment|size|tags",
            },
        )

        pages = rev_query.get("query", {}).get("pages", {})
        if not pages:
            raise ValueError(f"Revision {rev_id} not found on {wiki_domain}")

        page_data = next(iter(pages.values()))
        if "missing" in page_data or "badrevids" in rev_query.get("query", {}):
            raise ValueError(f"Revision {rev_id} does not exist on {wiki_domain}")

        revisions = page_data.get("revisions", [])
        if not revisions:
            raise ValueError(f"No revision payload returned for {rev_id} on {wiki_domain}")

        rev_info = revisions[0]
        parent_id = int(rev_info.get("parentid") or 0)
        title = page_data.get("title", "")
        page_id = int(page_data.get("pageid") or 0)
        user = rev_info.get("user", "Unknown")
        timestamp = rev_info.get("timestamp", "")
        comment = rev_info.get("comment", "") or ""
        tags = rev_info.get("tags", []) or []
        size = int(rev_info.get("size") or 0)

        diff_html = ""
        if parent_id > 0:
            compare_data = await _api_get(
                client,
                wiki_domain,
                {
                    "action": "compare",
                    "fromrev": str(parent_id),
                    "torev": str(rev_id),
                    "prop": "diff|user|comment|title|timestamp|size|ids",
                },
            )
            compare_obj = compare_data.get("compare", {})
            diff_html = compare_obj.get("*", "") or ""
            parsed_diff = _parse_compare_html(diff_html)
            title = compare_obj.get("totitle") or title
            comment = compare_obj.get("tocomment") if compare_obj.get("tocomment") is not None else comment
            user = compare_obj.get("touser") or user
            timestamp = compare_obj.get("totimestamp") or timestamp
        else:
            # Initial page creation (parentid == 0): fetch revision wikitext directly
            content_query = await _api_get(
                client,
                wiki_domain,
                {
                    "action": "query",
                    "prop": "revisions",
                    "revids": str(rev_id),
                    "rvprop": "content",
                    "rvslots": "main",
                },
            )
            c_pages = content_query.get("query", {}).get("pages", {})
            c_page = next(iter(c_pages.values()), {})
            c_revs = c_page.get("revisions", [{}])
            raw_text = c_revs[0].get("slots", {}).get("main", {}).get("*", "") if c_revs else ""
            added_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            parsed_diff = {
                "added_lines": added_lines,
                "deleted_lines": [],
                "added_highlights": added_lines,
                "deleted_highlights": [],
                "added_text": "\n".join(added_lines),
                "deleted_text": "",
                "added_chars": len("\n".join(added_lines)),
                "deleted_chars": 0,
            }

        added_urls_all, _ = extract_urls_and_domains(parsed_diff["added_text"])
        deleted_urls_all, _ = extract_urls_and_domains(parsed_diff["deleted_text"])
        deleted_url_set = set(deleted_urls_all)
        newly_added_urls = [u for u in added_urls_all if u not in deleted_url_set]

        newly_added_domains: list[str] = []
        seen_doms: set[str] = set()
        for url_str in newly_added_urls:
            dom = normalize_domain(url_str)
            if dom and dom not in seen_doms:
                seen_doms.add(dom)
                newly_added_domains.append(dom)

        return {
            "wiki_domain": wiki_domain,
            "wiki": domain_to_wiki(wiki_domain),
            "rev_id": int(rev_id),
            "parent_id": parent_id,
            "page_id": page_id,
            "title": title,
            "user": user,
            "timestamp": timestamp,
            "comment": comment,
            "size": size,
            "tags": tags,
            "is_reverted": "mw-reverted" in tags,
            "diff_url": f"https://{wiki_domain}/w/index.php?diff={rev_id}&oldid={parent_id}"
            if parent_id
            else f"https://{wiki_domain}/w/index.php?oldid={rev_id}",
            "added_lines": parsed_diff["added_lines"],
            "deleted_lines": parsed_diff["deleted_lines"],
            "added_highlights": parsed_diff["added_highlights"],
            "deleted_highlights": parsed_diff["deleted_highlights"],
            "added_text": parsed_diff["added_text"],
            "deleted_text": parsed_diff["deleted_text"],
            "added_chars": parsed_diff["added_chars"],
            "deleted_chars": parsed_diff["deleted_chars"],
            "added_urls": newly_added_urls,
            "added_domains": newly_added_domains,
            "diff_html": diff_html,
        }
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def fetch_user_profile(
    wiki_domain: str,
    username: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch editor editcount, registration timestamp, account age in days, and groups."""
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    try:
        assert client is not None
        if IPV4_REGEX.match(username) or (":" in username and IPV6_REGEX.match(username)):
            return {
                "username": username,
                "wiki_domain": wiki_domain,
                "editcount": 0,
                "registration": None,
                "age_days": 0.0,
                "groups": ["anonymous"],
                "is_anonymous": True,
                "is_temp": False,
            }

        data = await _api_get(
            client,
            wiki_domain,
            {
                "action": "query",
                "list": "users",
                "ususers": username,
                "usprop": "editcount|registration|groups",
            },
        )
        users = data.get("query", {}).get("users", [])
        if not users:
            return {
                "username": username,
                "wiki_domain": wiki_domain,
                "editcount": 0,
                "registration": None,
                "age_days": 0.0,
                "groups": [],
                "is_anonymous": True,
                "is_temp": username.startswith("~"),
            }

        u = users[0]
        if "invalid" in u or "missing" in u:
            return {
                "username": username,
                "wiki_domain": wiki_domain,
                "editcount": 0,
                "registration": None,
                "age_days": 0.0,
                "groups": ["anonymous"],
                "is_anonymous": True,
                "is_temp": username.startswith("~"),
            }

        groups = u.get("groups", []) or []
        editcount = int(u.get("editcount") or 0)
        reg_str = u.get("registration")
        age_days = 0.0
        if reg_str:
            try:
                reg_dt = datetime.fromisoformat(reg_str.replace("Z", "+00:00"))
                age_days = max(0.0, (datetime.now(timezone.utc) - reg_dt).total_seconds() / 86400.0)
            except ValueError:
                age_days = 0.0
        elif editcount > 0:
            # Very old pre-2006 accounts on Wikipedia sometimes have null registration
            age_days = 3650.0

        is_temp = ("temp" in groups) or username.startswith("~")
        return {
            "username": u.get("name", username),
            "wiki_domain": wiki_domain,
            "editcount": editcount,
            "registration": reg_str,
            "age_days": round(age_days, 2),
            "groups": groups,
            "is_anonymous": False,
            "is_temp": is_temp,
        }
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def stream_recent_changes(stop_event: asyncio.Event | None = None) -> AsyncIterator[dict[str, Any]]:
    """Yield filtered live edit events from Wikimedia EventStreams SSE.

    Filters for:
    - type == "edit"
    - namespace == 0 (Main / Article namespace)
    - bot == False
    - wiki in settings.wiki_stream_wikis
    """
    backoff = 2.0
    while stop_event is None or not stop_event.is_set():
        try:
            logger.info(
                "Connecting to Wikimedia EventStreams SSE url=%s wikis=%s",
                settings.wiki_stream_url,
                sorted(settings.wiki_stream_wikis),
            )
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                async with client.stream(
                    "GET",
                    settings.wiki_stream_url,
                    headers={
                        "User-Agent": settings.wiki_user_agent,
                        "Accept": "text/event-stream",
                    },
                ) as response:
                    response.raise_for_status()
                    backoff = 2.0
                    data_lines: list[str] = []

                    async for line in response.aiter_lines():
                        if stop_event is not None and stop_event.is_set():
                            return

                        if line == "":
                            if data_lines:
                                payload_str = "\n".join(data_lines)
                                data_lines.clear()
                                try:
                                    event = json.loads(payload_str)
                                except json.JSONDecodeError:
                                    continue

                                if (
                                    event.get("type") == "edit"
                                    and event.get("namespace") == 0
                                    and event.get("bot") is False
                                    and event.get("wiki") in settings.wiki_stream_wikis
                                ):
                                    yield event
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())

        except asyncio.CancelledError:
            logger.info("Wikimedia EventStreams consumer cancelled")
            raise
        except Exception as exc:
            logger.warning("Wikimedia EventStreams error (%s); reconnecting in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)
