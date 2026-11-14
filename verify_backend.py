"""End-to-end verification script for the WikiProof FastAPI backend.

Starts or probes the FastAPI application via httpx ASGI transport / live server
and verifies `/api/health`, `/api/presets`, `/api/inspect` on a real Wikipedia
revision diff URL, `/api/network`, `/api/benchmark` (including dynamic threshold
recalculation), and `/api/live-feed`.
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

from main import app, lifespan


async def verify_all() -> None:
    """Run end-to-end verification checks against the WikiProof FastAPI app."""
    async with lifespan(app):
        # Allow background Wikimedia SSE worker a brief window to ingest live edits
        await asyncio.sleep(5.0)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as client:
            # 0. Frontend dashboard HTML
            r_root = await client.get("/")
            assert r_root.status_code == 200, f"GET / failed: {r_root.status_code}"
            assert "WikiProof" in r_root.text and "vis-network" in r_root.text
            print("[PASS] GET / -> served frontend/index.html (", len(r_root.text), "bytes)")

            # 1. Healthcheck
            r_health = await client.get("/api/health")
            assert r_health.status_code == 200, f"/api/health failed: {r_health.text}"
            health = r_health.json()
            assert health["status"] == "ok"
            print("[PASS] GET /api/health ->", json.dumps(health, indent=2))

            # 2. Presets (3 real cached examples)
            r_presets = await client.get("/api/presets")
            assert r_presets.status_code == 200, f"/api/presets failed: {r_presets.text}"
            presets = r_presets.json()
            assert presets["count"] == 3, f"Expected 3 presets, got {presets['count']}"
            print(
                "[PASS] GET /api/presets ->",
                [(p["preset_key"], p["rev_id"], p["heuristics"]["score"], p["llm_analysis"]["verdict"]) for p in presets["presets"]],
            )

            # 3. Real revision diff inspection via Wikipedia URL
            inspect_url = "https://en.wikipedia.org/w/index.php?diff=1385606898&oldid=1385606435"
            r_inspect = await client.post("/api/inspect", json={"url": inspect_url, "force_refresh": True})
            assert r_inspect.status_code == 200, f"/api/inspect failed: {r_inspect.text}"
            dossier = r_inspect.json()
            assert dossier["rev_id"] == 1385606898
            assert dossier["llm_analysis"]["verdict"] in {"Suspicious", "Uncertain", "Benign"}
            assert isinstance(dossier["heuristics"]["score"], int)
            print(
                "[PASS] POST /api/inspect ->",
                f"title={dossier['title']} score={dossier['heuristics']['score']} "
                f"verdict={dossier['llm_analysis']['verdict']} domains={dossier['diff']['added_domains']}",
            )

            # 4. Network graph
            r_net = await client.get("/api/network")
            assert r_net.status_code == 200, f"/api/network failed: {r_net.text}"
            net = r_net.json()
            assert net["stats"]["total_nodes"] > 0 and net["stats"]["total_edges"] > 0
            print("[PASS] GET /api/network ->", net["stats"])

            # 5. Benchmark with dynamic threshold recalculation
            r_bench_50 = await client.get("/api/benchmark?threshold=50")
            r_bench_30 = await client.get("/api/benchmark?threshold=30")
            assert r_bench_50.status_code == 200 and r_bench_30.status_code == 200
            b50 = r_bench_50.json()
            b30 = r_bench_30.json()
            assert b50["total_cases"] >= 10
            print("[PASS] GET /api/benchmark?threshold=50 ->", b50["confusion_matrix"], b50["metrics"])
            print("[PASS] GET /api/benchmark?threshold=30 ->", b30["confusion_matrix"], b30["metrics"])

            # 6. Live feed
            r_live = await client.get("/api/live-feed?limit=10")
            assert r_live.status_code == 200, f"/api/live-feed failed: {r_live.text}"
            live = r_live.json()
            print(f"[PASS] GET /api/live-feed -> count={live['count']}")
            if live["events"]:
                sample = live["events"][0]
                print(
                    "       Sample live event:",
                    f"wiki={sample['wiki']} rev={sample['rev_id']} title={sample['title']} score={sample['risk_score']}",
                )


if __name__ == "__main__":
    try:
        asyncio.run(verify_all())
        print("\nALL BACKEND VERIFICATION CHECKS PASSED.")
    except Exception as exc:
        print(f"\nVERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
