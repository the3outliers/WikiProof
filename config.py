"""Centralized configuration and logging for the WikiProof forensic engine.

Loads environment variables from `.env` (if present), defines Wikimedia stream
targets, curated high-reliability reference domains, risk scoring thresholds,
and LLM router settings without ever hardcoding or logging secrets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_set(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return {item.strip() for item in raw.split(",") if item.strip()}


def _clean_url(name: str, default: str) -> str:
    raw = os.getenv(name, default).strip()
    # Unwrap accidental Markdown links like [https://example.com](https://example.com)
    if raw.startswith("[") and "](" in raw and raw.endswith(")"):
        return raw.split("](", 1)[1][:-1].strip()
    return raw


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings loaded from environment variables."""

    hostinger_ai_key: str = field(default_factory=lambda: os.getenv("HOSTINGER_AI_KEY", "").strip())
    hostinger_base_url: str = field(
        default_factory=lambda: _clean_url("HOSTINGER_BASE_URL", "https://router.hostinger.com/v1")
    )
    hostinger_model_slug: str = field(
        default_factory=lambda: os.getenv("HOSTINGER_MODEL_SLUG", "openai/gpt-4o-mini").strip()
    )

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model_slug: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_SLUG", "gemini-4-hydrogen").strip()
    )

    wiki_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "WIKI_USER_AGENT",
            "WikiProof-Junction2026/1.0 (https://wikiproof.live; team@wikiproof.live)",
        ).strip()
    )
    wiki_stream_url: str = "https://stream.wikimedia.org/v2/stream/recentchange"
    wiki_stream_wikis: set[str] = field(
        default_factory=lambda: _env_set("WIKI_STREAM_WIKIS", {"enwiki", "fiwiki", "dewiki", "ruwiki"})
    )

    risk_threshold_default: int = field(default_factory=lambda: _env_int("RISK_THRESHOLD_DEFAULT", 50))
    risk_threshold_uncertain: int = field(default_factory=lambda: _env_int("RISK_THRESHOLD_UNCERTAIN", 30))
    risk_threshold_suspicious: int = field(default_factory=lambda: _env_int("RISK_THRESHOLD_SUSPICIOUS", 60))
    llm_auto_analyze_threshold: int = field(default_factory=lambda: _env_int("LLM_AUTO_ANALYZE_THRESHOLD", 30))
    cross_wiki_window_minutes: int = field(default_factory=lambda: _env_int("CROSS_WIKI_WINDOW_MINUTES", 60))

    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/wikiproof.db").strip())
    benchmark_results_path: str = field(
        default_factory=lambda: os.getenv("BENCHMARK_RESULTS_PATH", "data/benchmark_results.json").strip()
    )

    stream_enabled: bool = field(default_factory=lambda: _env_bool("STREAM_ENABLED", True))
    backend_host: str = field(default_factory=lambda: os.getenv("BACKEND_HOST", "0.0.0.0").strip())
    backend_port: int = field(default_factory=lambda: _env_int("BACKEND_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip().upper())

    def has_valid_hostinger_key(self) -> bool:
        """Return True when HOSTINGER_AI_KEY is present and not a placeholder."""
        if not self.hostinger_ai_key:
            return False
        placeholders = {"YOUR_KEY_HERE", "sk-hstgr-YOUR_KEY_HERE", "MY_HOSTINGER_AI_KEY"}
        return self.hostinger_ai_key not in placeholders and not self.hostinger_ai_key.endswith("YOUR_KEY_HERE")

    def has_valid_gemini_key(self) -> bool:
        """Return True when GEMINI_API_KEY is present and not a placeholder."""
        if not self.gemini_api_key:
            return False
        return self.gemini_api_key not in {"MY_GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"}


settings = Settings()


def configure_logging() -> None:
    """Configure root logging format once."""
    level = getattr(logging, settings.log_level, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger."""
    configure_logging()
    return logging.getLogger(name)
