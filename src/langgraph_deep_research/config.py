"""Configuration: load settings from environment variables and .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    tavily_api_key: str
    model: str


def load_settings() -> Settings:
    """Return current settings; env vars override .env."""
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        model=os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b"),
    )
