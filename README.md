# WikiProof

**Detecting coordinated Wikipedia citation-laundering rings and sockpuppet narrative manipulation before poisoned edits pollute AI training data.**

WikiProof monitors public Wikipedia edits and surfaces behavioral patterns that may indicate citation laundering or narrative manipulation. It combines transparent risk signals, revision diffs, and AI-assisted analysis to help editors and researchers prioritize human review.

> WikiProof is a triage and research tool. A score or AI verdict is not proof of sockpuppetry, coordination, or intent, and the system does not block edits or control downstream AI datasets.

## What it does

1. **Streams live edits.** Connects to Wikimedia EventStreams and fetches revision diffs and editor metadata through the MediaWiki API. The default language editions are English, Finnish, German, Russian, and Swedish; configure the list with `WIKI_STREAM_WIKIS`.
2. **Scores behavioral signals.** Checks account age and edit count, newly added external domains, substantial content or citation removals, possible username/article overlap, promotional wording, and matching topic or domain activity across language editions.
3. **Explains revision changes.** Sends the added and removed text to the configured AI provider for a plain-English verdict and explanation. Non-English revisions also include an AI-generated English summary of the changes when available. The original diff remains available for verification.
4. **Assesses propaganda.** The AI verdict includes a propaganda assessment: whether the change pushes a political, state, or ideological narrative in Wikipedia's voice, which techniques it uses (whitewashing, false equivalence, euphemism, fake sourcing, demonization, historical revisionism), and whom the framing favors. Edits touching politically sensitive topics (`topics.py`) are sent to this review even when their behavioral risk score is low. Flagged edits are marked in the live feed and listed at `GET /api/propaganda`.
5. **Supports calibration.** Provides benchmark cases and adjustable thresholds with precision, recall, F1, false-positive rate, and confusion-matrix metrics.

The dashboard includes a live edit radar, revision inspector, editor–article–domain relationship graph, and benchmark calibration view.

## False positives and limitations

- Veteran editors, high-reliability and government/academic sources, routine maintenance summaries, and small edits can reduce a risk score.
- Cross-language topic or domain matches are signals for review; they do not establish that editors are coordinating.
- Domain reputation checks use configured heuristics and a curated list, not a complete or authoritative source-quality assessment.
- Revert handling is based on edit-summary cues; it is not a full reconstruction of revert history.
- Benchmark metrics describe the cases in the local dataset and should not be treated as performance guarantees for all Wikipedia edits.
- AI-generated verdicts and English summaries can be wrong. Check the original revision, cited sources, and editor history before drawing conclusions.

## Technology

- **Backend:** Python, FastAPI, and Uvicorn
- **Storage and relationship data:** SQLite
- **Sources:** Wikimedia EventStreams and MediaWiki Action API
- **Analysis:** deterministic heuristics; Hostinger AI Router as the primary LLM provider, with optional Google Gemini fallback
- **Dashboard:** HTML, Tailwind CSS, Alpine.js, and vis-network
- **Deployment configuration:** Docker Compose and Caddy for a VPS deployment

When AI analysis is enabled, revision text and associated analysis context are sent to the configured AI provider. Configure credentials locally and do not commit them.

## Run locally

### Prerequisites

- Python 3.10 or newer
- Node.js and npm

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm install --legacy-peer-deps
Copy-Item .env.example .env
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Edit `.env` to add `HOSTINGER_AI_KEY` for Hostinger AI Router analysis. `GEMINI_API_KEY` is optional and used as a fallback. Without either key, the app still runs with deterministic heuristic analysis.

Start the app:

```bash
npm run dev -- --host 127.0.0.1 --port 3000
```

Open [http://localhost:3000](http://localhost:3000). The health endpoint is available at [http://localhost:3000/api/health](http://localhost:3000/api/health).

## Configuration

See `.env.example` for the available settings. Common options:

| Variable | Purpose |
| --- | --- |
| `HOSTINGER_AI_KEY` | Hostinger AI Router key |
| `GEMINI_API_KEY` | Optional Google Gemini fallback key |
| `WIKI_STREAM_WIKIS` | Comma-separated Wikimedia wiki IDs to monitor |
| `STREAM_ENABLED` | Enable or disable the background EventStreams worker |
| `RISK_THRESHOLD_DEFAULT` | Default benchmark threshold |
| `DB_PATH` | SQLite database location |

## API

- `GET /api/health` — service, stream, database, and AI-provider status
- `GET /api/live-feed` — recent scored edits; supports `limit`, `wiki`, `min_score`, and `propaganda=true`
- `GET /api/propaganda` — analyzed edits whose AI assessment detected a propaganda narrative
- `POST /api/inspect` — inspect a Wikipedia diff URL or `{ "wiki_domain": "...", "rev_id": 123 }`
- `GET /api/presets` — cached forensic examples
- `GET /api/network` — editor, article, and external-domain graph data
- `GET /api/benchmark` — benchmark metrics and threshold sweep

See [Design.md](./Design.md) for the architecture and scoring details.

## Roadmap

- Webhook alerts for WikiProjects, editors, and journalists
- A citation-integrity API for downstream AI search and retrieval systems
