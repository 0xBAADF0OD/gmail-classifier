"""Environment-backed configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKENDS = ("jev", "llm", "rules")


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    value = env.get(key, default)
    return value.strip() if value else default


def _is_placeholder(value: str) -> bool:
    return not value or value.lower().startswith(("sk-placeholder", "changeme", "replace-me"))


@dataclass(frozen=True)
class Config:
    backend: str
    typesafe_api_key: str
    typesafe_model: str
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    credentials_file: Path
    token_file: Path
    user_id: str
    notes: tuple[str, ...] = ()

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

        typesafe_key = _get(values, "TYPESAFE_API_KEY")
        llm_base = _get(values, "LLM_API_BASE", "http://localhost:4000/v1")
        llm_key = _get(values, "LLM_API_KEY")
        llm_model = _get(values, "LLM_MODEL", "pegasus-lite")

        requested = (backend or _get(values, "CLASSIFIER_BACKEND", "jev")).lower()
        if requested not in BACKENDS:
            raise ValueError(f"unknown backend {requested!r}; choose one of {', '.join(BACKENDS)}")

        notes: list[str] = []
        resolved = requested
        if requested == "jev" and not typesafe_key:
            if backend is None:
                resolved = "llm"
                notes.append(
                    "TYPESAFE_API_KEY is empty - falling back to the OpenAI-compatible LLM backend."
                )
            else:
                raise ValueError(
                    "backend 'jev' needs TYPESAFE_API_KEY (get one at https://console.typesafe.ai/keys)"
                )
        if resolved == "llm" and _is_placeholder(llm_key):
            notes.append(f"LLM_API_KEY looks like a placeholder; {llm_base} may reject requests")

        return cls(
            backend=resolved,
            typesafe_api_key=typesafe_key,
            typesafe_model=_get(values, "TYPESAFE_MODEL", "jev-latest"),
            llm_api_base=llm_base,
            llm_api_key=llm_key or "sk-placeholder",
            llm_model=llm_model,
            credentials_file=Path(
                _get(values, "GMAIL_CREDENTIALS_FILE", "credentials.json")
            ).expanduser(),
            token_file=Path(_get(values, "GMAIL_TOKEN_FILE", "token.json")).expanduser(),
            user_id=_get(values, "GMAIL_USER_ID", "me"),
            notes=tuple(notes),
        )
