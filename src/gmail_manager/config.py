"""Environment-backed configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKENDS = ("llm", "rules")


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    value = env.get(key, default)
    return value.strip() if value else default


def _is_placeholder(value: str) -> bool:
    return not value or value.lower().startswith(("sk-placeholder", "changeme", "replace-me"))


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    backend: str
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    credentials_file: Path
    token_file: Path
    user_id: str
    notes: tuple[str, ...] = ()
    debug: bool = False

    @classmethod
    def from_env(
        cls,
        backend: str | None = None,
        *,
        dotenv: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> Config:
        if dotenv:
            load_dotenv(override=False)
        source: Mapping[str, str] = os.environ if env is None else env
        values = {str(key): str(value) for key, value in source.items()}

        llm_base = _get(values, "LLM_API_BASE", "http://localhost:4000/v1")
        llm_key = _get(values, "LLM_API_KEY")
        llm_model = _get(values, "LLM_MODEL", "pegasus-lite")

        requested = (backend or _get(values, "CLASSIFIER_BACKEND", "llm")).lower()
        if requested not in BACKENDS:
            raise ValueError(f"unknown backend {requested!r}; choose one of {', '.join(BACKENDS)}")

        notes: list[str] = []
        if requested == "llm" and _is_placeholder(llm_key):
            notes.append(f"LLM_API_KEY looks like a placeholder; {llm_base} may reject requests")

        debug = _is_truthy(_get(values, "GMAIL_MANAGER_DEBUG"))

        return cls(
            backend=requested,
            llm_api_base=llm_base,
            llm_api_key=llm_key or "sk-placeholder",
            llm_model=llm_model,
            credentials_file=Path(
                _get(values, "GMAIL_CREDENTIALS_FILE", "credentials.json")
            ).expanduser(),
            token_file=Path(_get(values, "GMAIL_TOKEN_FILE", "token.json")).expanduser(),
            user_id=_get(values, "GMAIL_USER_ID", "me"),
            notes=tuple(notes),
            debug=debug,
        )
