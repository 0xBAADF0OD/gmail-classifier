from __future__ import annotations

import pytest

from gmail_manager.config import Config

BASE = {"GMAIL_CREDENTIALS_FILE": "client.json", "GMAIL_TOKEN_FILE": "tok.json"}


def make(env: dict[str, str], backend: str | None = None) -> Config:
    return Config.from_env(backend, dotenv=False, env={**BASE, **env})


def test_llm_is_the_default_backend() -> None:
    cfg = make({"LLM_API_KEY": "sk-ab12cd34"})
    assert cfg.backend == "llm"
    assert cfg.notes == ()


def test_rules_backend_selectable() -> None:
    cfg = make({"CLASSIFIER_BACKEND": "rules"})
    assert cfg.backend == "rules"


def test_placeholder_llm_key_warns() -> None:
    cfg = make({"LLM_API_KEY": "sk-placeholder-replace-me", "LLM_API_BASE": "http://proxy:4000/v1"})
    assert cfg.backend == "llm"
    assert any("placeholder" in note for note in cfg.notes)


def test_missing_llm_key_warns() -> None:
    cfg = make({})
    assert any("LLM_API_KEY looks like a placeholder" in note for note in cfg.notes)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        make({}, backend="crystal-ball")


def test_explicit_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        make({"CLASSIFIER_BACKEND": "crystal-ball"})


def test_debug_flag_reads_env() -> None:
    assert make({"GMAIL_MANAGER_DEBUG": "1"}).debug is True
    assert make({"GMAIL_MANAGER_DEBUG": "off"}).debug is False
    assert make({}).debug is False


def test_paths_and_model_defaults() -> None:
    cfg = make({})
    assert str(cfg.credentials_file) == "client.json"
    assert cfg.user_id == "me"
    assert cfg.llm_model == "pegasus-lite"
    assert cfg.llm_api_base == "http://localhost:4000/v1"
