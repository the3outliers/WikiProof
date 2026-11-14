# WikiProof — Architecture & Forensic Engine Design

WikiProof is a real-time Wikipedia edit intelligence and covert influence-operation detection engine built in Python (FastAPI). Every revision, diff, user profile, live stream event, and benchmark case is fetched from live Wikimedia APIs and cached locally in SQLite.

## System Architecture

```
Wikimedia EventStreams SSE (enwiki, fiwiki, dewiki, ruwiki)
        │  [type=edit, ns=0, bot=false]
        ▼
  wikimedia_client.py  ◄── MediaWiki Action API (query revisions + compare diff + users)
        │
        ▼
   heuristics.py       ──► Deterministic Risk Score (0-100) + False-Positive Guardrails
        │
        ├──► database.py (SQLite WAL: data/wikiproof.db)
        │       ├── live_events
        │       ├── revisions
        │       ├── network_edges (Editor -> Article -> External Domain)
        │       └── benchmark_cases
        │
        └──► llm_analyzer.py
                ├── Primary: Hostinger AI Router (OpenAI SDK)
                └── Fallback: Google GenAI SDK (gemini-4-hydrogen)
```

## Module Responsibilities

| File | Concern |
| :--- | :--- |
| `config.py` | Environment variable loading, API/router config, risk thresholds, structured logger (no secrets logged). |
| `wikimedia_client.py` | EventStreams SSE consumer, Wikipedia diff URL parser, MediaWiki `action=query` + `action=compare` diff extractor (`ins.diffchange`/`td.diff-addedline` and `del.diffchange`/`td.diff-deletedline`), URL/domain extraction, and `list=users` profile fetcher. |
| `heuristics.py` | Deterministic behavioral risk scorer (`0–100`) combining risk signals and explicit false-positive guardrails. |
| `llm_analyzer.py` | Strict-JSON forensic LLM analyzer using Hostinger AI Router (`HOSTINGER_BASE_URL`) with automatic fallback to Google GenAI SDK (`GEMINI_API_KEY`). |
| `database.py` | SQLite persistence (`data/wikiproof.db`), cross-language burst lookup, network graph builder, and dynamic confusion matrix computation. |
| `seed_benchmark.py` | Fetches real historical Wikipedia revisions (documented reverted/COI/link-injection edits vs. clean Featured Article volunteer edits), evaluates them end-to-end, stores results in SQLite + `data/benchmark_results.json`, and prints Precision, Recall, F1, FPR, and Confusion Matrix. |
| `main.py` | FastAPI server managing the background SSE worker and REST endpoints. |

## Scoring Logic (`heuristics.py`)

### Behavioral Risk Signals (Additive)
1. **Single-Purpose / New / Temporary Account** (`+25` to `+30`): Unregistered IP, temporary account (`~2026-...`), or registered account with `age_days < 30` or `editcount < 50`.
2. **Unverified External Link Injection** (`+25` to `+35`): Newly added external URLs whose domain is outside the curated high-reliability whitelist (`reuters.com`, `apnews.com`, `bbc.com`, `yle.fi`, `nature.com`, `doi.org`, `europa.eu`, academic/government domains).
3. **Large Content / Citation Deletion** (`+25` to `+32`): `deleted_chars > 300` with minimal replacement (`added_chars < max(120, 0.35 * deleted_chars)`), weighted higher when `<ref>`/`cite`/URLs are removed.
4. **Cross-Language Burst Detection** (`+25`): Same external domain or normalized article topic edited across `>= 2` language wikis within `60` minutes.
5. **Supporting Signals**: Potential COI username/article title overlap (`+20`), promotional peacock phrasing (`+15`), unsourced large addition by low-edit user (`+12`).

### False-Positive Guardrails (Subtractive)
1. **Veteran Editor Guardrail** (`-25` to `-30`): `editcount > 2000` and `age_days > 365`.
2. **Official Government / Academic & High-Reliability Citations** (`-15` to `-25`): Automatically recognizes any domain ending in `.gov`, `.gov.uk`, `.gc.ca`, `.gov.bc.ca`, `.edu`, `.ac.uk`, or `.europa.eu` (`-25` when no unverified domains are present, `-15` alongside unverified links) or curated high-reliability domains (`-20`).
3. **Routine Maintenance / Anti-Vandalism Summary** (`-20`): Edit summary matches maintenance/reversion terms (`typo`, `copyedit`, `rvv`, `revert`, `vandalism`, etc.) with no unverified links.
4. **Minor Formatting Adjustment** (`-10`): `<= 40` chars added/deleted with no external links.

Final score is clamped to `[0, 100]`.

## LLM Forensic Output Schema (`llm_analyzer.py`)

- `verdict`: `"Suspicious" | "Uncertain" | "Benign"`
- `confidence`: `float` (`0.0–1.0`)
- `techniques`: subset of `["Whitewashing", "Loaded Framing", "Fabricated Authority", "Source Laundering"]` (always `[]` if `Benign`)
- `highlighted_quote`: exact verbatim substring from the diff driving the verdict
- `explanation`: 2-sentence plain-English explanation for an everyday reader

## REST & UI Endpoints (`main.py`)

- `GET /` — Serves `frontend/index.html` (Tailwind CSS CDN, Alpine.js, `vis-network` CDN; zero Node build step).
- `GET /api/health` — Service health, SSE stream stats, SQLite counts, and LLM router status.
- `GET /api/live-feed?limit=50&wiki=enwiki&min_score=0` — Latest scored edits from the Wikimedia SSE stream.
- `POST /api/inspect` — Accepts `{ "url": "..." }` or `{ "wiki_domain": "...", "rev_id": ... }`, runs full pipeline, caches and returns the forensic dossier.
- `GET /api/presets` — Returns 3 cached real examples (`true_positive`, `hard_negative`, `uncertain`).
- `GET /api/network?min_score=0&limit=120` — Returns `Editor -> Article -> External Domain` graph nodes and edges.
- `GET /api/benchmark?threshold=50` — Returns benchmark cases, metrics, threshold sweep, and dynamically recomputed confusion matrix.

## Frontend Dashboard (`frontend/index.html`)

- **Live Radar**: Auto-refreshing feed of EN/FI/DE/RU Wikipedia edits with behavioral risk badges (`Benign`, `Uncertain`, `Suspicious`), triggered signals, and one-click **Inspect with AI**.
- **Revision Inspector**: Diff URL / revision ID sandbox, 3 one-click real presets, side-by-side color-highlighted `−`/`+` diff, signals vs. guardrails breakdown, and AI verdict card.
- **Citation & Editor Graph**: Interactive `vis-network` topology (`Editor -> Article -> External Domain`) with multi-editor domain cluster detection.
- **Calibration & Precision**: Live Precision, Recall, F1, FPR, 2×2 Confusion Matrix, and interactive Risk Threshold Slider (`/api/benchmark?threshold=...`).

## Production Deployment (Ubuntu VPS with Docker & Caddy)

```bash
# 1. Install Docker & Docker Compose plugin on Ubuntu 24.04 / 22.04
sudo apt-get update && sudo apt-get install -y ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# 2. Configure environment & persistent data directory
cp .env.example .env
# Edit .env to set HOSTINGER_AI_KEY, optional GEMINI_API_KEY, and WIKIPROOF_DOMAIN=your-domain.com
nano .env
mkdir -p data

# 3. Pre-seed real Wikipedia benchmark dataset into ./data/wikiproof.db
docker compose run --rm wikiproof python seed_benchmark.py

# 4. Launch Uvicorn (port 8000 internal) + Caddy (80/443 automatic HTTPS)
docker compose up -d --build

# 5. Inspect logs & verify live health
docker compose logs -f wikiproof caddy
curl -fsS http://localhost/api/health
```

## Local Verification

```bash
# 1. Seed real benchmark dataset & print metrics
.venv/bin/python seed_benchmark.py

# 2. Run end-to-end verification suite
.venv/bin/python verify_backend.py

# 3. Start FastAPI dashboard & API locally
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

