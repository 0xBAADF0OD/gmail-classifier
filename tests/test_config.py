from __future__ import annotations

import pytest

from gmail_manager.config import Config

BASE = {"GMAIL_CREDENTIALS_FILE": "client.json", "GMAIL_TOKEN_FILE": "tok.json"}


def make(env: dict[str, str], backend: str | None = None) -> Config:
    return Config.from_env(backend, dotenv=False, env={**BASE, **env})


def test_jev_without_key_falls_back_to_llm() -> None:
    cfg = make({"TYPESAFE_API_KEY": ""})
    assert cfg.backend == "llm"
    assert any("TYPESAFE_API_KEY is empty" in note for note in cfg.notes)


def test_jev_used_when_key_present() -> None:
    cfg = make({"TYPESAFE_API_KEY": "tsK_live", "TYPESAFE_MODEL": "jev-1.13.0"})
    assert cfg.backend == "jev"
    assert cfg.typesafe_model == "jev-1.13.0"
    assert cfg.notes == ()


def test_explicit_jev_without_key_raises() -> None:
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        make({"TYPESAFE_API_KEY": ""}, backend="jev")


def test_placeholder_llm_key_warns() -> None:
    cfg = make({"LLM_API_KEY": "sk-placeholder-replace-me", "LLM_API_BASE": "http://proxy:4000/v1"})
    assert cfg.backend == "llm"
    assert any("placeholder" in note for note in cfg.notes)


def test_real_llm_key_does_not_warn() -> None:
    cfg = make({"LLM_API_KEY": "sk-ab12cd34"}, backend="llm")
    assert cfg.notes == ()


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        make({}, backend="crystal-ball")


def test_paths_and_model_defaults() -> None:
    cfg = make({})
    assert str(cfg.credentials_file) == "client.json"
    assert cfg.user_id == "me"
    assert cfg.llm_model == "pegasus-lite"
