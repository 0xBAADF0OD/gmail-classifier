from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeGmail, raw_message
from gmail_manager import cli
from gmail_manager.cli import main

MESSAGES = {
    "m_otp": raw_message(
        "m_otp",
        sender="GitHub <noreply@github.com>",
        subject="Your GitHub verification code is 771203",
        body="Use 771203 to sign in to GitHub.",
    ),
    "m_receipt": raw_message(
        "m_receipt",
        sender="Amazon <auto-confirm@amazon.com>",
        subject="Your receipt for order 112-99",
        body="Thanks for your purchase. Your receipt is attached.",
    ),
    "m_sale": raw_message(
        "m_sale",
        sender="Brand <offers@brand.com>",
        subject="48 hours only: 40% off",
        body="Limited time offer, shop now.",
    ),
}


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeGmail:
    fake = FakeGmail({key: dict(value) for key, value in MESSAGES.items()})
    monkeypatch.setattr(cli, "get_service", lambda *args, **kwargs: fake)
    return fake


def run(*args: str) -> Any:
    return CliRunner().invoke(main, args, catch_exceptions=False)


def use(monkeypatch: pytest.MonkeyPatch, fake: FakeGmail) -> None:
    monkeypatch.setattr(cli, "get_service", lambda *args, **kwargs: fake)


def test_classify_dry_run_does_not_write(service: FakeGmail, clean_env: None) -> None:
    result = run("classify", "--backend", "rules", "--dry-run", "--confidence", "0.2")
    assert result.exit_code == 0
    assert service.modify_calls == []
    assert "would apply LL_OTP" in result.output
    assert "OTP=1" in result.output


def test_classify_applies_existing_labels(service: FakeGmail, clean_env: None) -> None:
    result = run("classify", "--backend", "rules", "--confidence", "0.2")
    assert result.exit_code == 0, result.output
    applied = {call["id"]: call["body"]["addLabelIds"] for call in service.modify_calls}
    assert applied == {
        "m_otp": ["LL_OTP"],
        "m_receipt": ["LL_TRANSACTION"],
        "m_sale": ["LL_PROMOTIONAL"],
    }


def test_classify_skips_already_labelled_messages(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    fake = FakeGmail(
        {
            "m_done": raw_message(
                "m_done",
                sender="Brand",
                subject="40% off",
                body="limited time offer",
                label_ids=["INBOX", "LL_PROMOTIONAL"],
            )
        }
    )
    use(monkeypatch, fake)
    result = run("classify", "--backend", "rules", "--confidence", "0.2")
    assert "nothing to classify (skipped 1 already labelled)" in result.output
    assert fake.modify_calls == []


def test_classify_reclassifies_on_request(monkeypatch: pytest.MonkeyPatch, clean_env: None) -> None:
    fake = FakeGmail(
        {
            "m_done": raw_message(
                "m_done",
                sender="Brand",
                subject="40% off",
                body="limited time offer",
                label_ids=["INBOX", "LL_PROMOTIONAL"],
            )
        }
    )
    use(monkeypatch, fake)
    result = run("classify", "--backend", "rules", "--confidence", "0.2", "--reclassify")
    assert "already-labelled=1" in result.output


def test_classify_honours_label_remap_and_warns_when_absent(
    service: FakeGmail, clean_env: None
) -> None:
    result = run(
        "classify",
        "--backend",
        "rules",
        "--dry-run",
        "--confidence",
        "0.2",
        "--label",
        "OTP=TwoFactor",
    )
    assert "no Gmail label for OTP -> 'TwoFactor'" in result.output
    assert "OTP=1" in result.output


def test_low_confidence_is_not_labelled(service: FakeGmail, clean_env: None) -> None:
    result = run("classify", "--backend", "rules", "--confidence", "0.9")
    assert "low-confidence=3" in result.output
    assert service.modify_calls == []


def test_classify_json_output(service: FakeGmail, clean_env: None) -> None:
    result = run("classify", "--backend", "rules", "--dry-run", "--confidence", "0.2", "--json")
    lines = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert len(lines) == 3
    otp = next(line for line in lines if line["messageId"] == "m_otp")
    assert otp["decision"]["category"] == "OTP"
    assert otp["label"] == "OTP"
    assert otp["outcome"] == "would-label"


def test_classify_rejects_bad_label_syntax(clean_env: None) -> None:
    result = CliRunner().invoke(main, ["classify", "--backend", "rules", "--label", "Spam=Junk"])
    assert result.exit_code != 0


def test_classify_reports_backend_errors(monkeypatch: pytest.MonkeyPatch, clean_env: None) -> None:
    fake = FakeGmail({"m_otp": MESSAGES["m_otp"]})
    use(monkeypatch, fake)

    class Broken:
        name = "broken"

        def classify(self, email: Any) -> Any:
            msg = "provider exploded"
            raise RuntimeError(msg)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "make_classifier", lambda *args, **kwargs: Broken())
    result = CliRunner().invoke(main, ["classify", "--backend", "rules"])
    assert result.exit_code == 1
    assert "provider exploded" in result.output


def test_classify_no_messages(monkeypatch: pytest.MonkeyPatch, clean_env: None) -> None:
    fake = FakeGmail({}, pages=[{"messages": []}])
    use(monkeypatch, fake)
    result = run("classify", "--backend", "rules")
    assert "nothing to classify" in result.output


def test_classify_uses_gmail_query(monkeypatch: pytest.MonkeyPatch, clean_env: None) -> None:
    fake = FakeGmail({}, pages=[{"messages": []}])
    use(monkeypatch, fake)
    run("classify", "--backend", "rules", "-q", "in:inbox is:unread", "-n", "7")
    assert fake.list_calls[0]["q"] == "in:inbox is:unread"
    assert fake.list_calls[0]["maxResults"] == 7


def test_classify_no_bodies_requests_metadata(service: FakeGmail, clean_env: None) -> None:
    run("classify", "--backend", "rules", "--dry-run", "--no-bodies", "--confidence", "0.2")
    assert service.get_calls[0]["format"] == "metadata"


def test_labels_command_maps_categories(service: FakeGmail, clean_env: None) -> None:
    result = run("labels")
    for category in ("OTP", "Primary", "Transaction", "Updates", "Promotional"):
        assert category in result.output


def test_labels_command_flags_missing(monkeypatch: pytest.MonkeyPatch, clean_env: None) -> None:
    fake = FakeGmail(labels=[{"id": "INBOX", "name": "INBOX", "type": "system"}])
    use(monkeypatch, fake)
    result = run("labels")
    assert "no label named" in result.output


def test_auth_check_without_token_file(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("GMAIL_TOKEN_FILE", str(tmp_path / "missing-token.json"))
    monkeypatch.setattr(cli, "get_service", lambda *args, **kwargs: FakeGmail())
    result = run("auth", "--check")
    assert result.exit_code != 0
    assert "no cached token" in result.output


def test_auth_reports_account(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("GMAIL_TOKEN_FILE", str(tmp_path / "token.json"))
    (tmp_path / "token.json").write_text("{}")
    monkeypatch.setattr(cli, "get_service", lambda *args, **kwargs: FakeGmail())
    result = run("auth", "--check")
    assert "authorized as me@example.com" in result.output
