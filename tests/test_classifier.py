from __future__ import annotations

from typing import Any

import pytest

from gmail_manager.classifier import LlmClassifier, RulesClassifier, parse_llm_answer
from gmail_manager.state import EmailText


def email(**kwargs: Any) -> EmailText:
    defaults: dict[str, Any] = {
        "message_id": "m1",
        "sender": "Amazon <auto-confirm@amazon.com>",
        "subject": "Your order has shipped",
        "body": "Your order #112 has shipped and arrives Tuesday.",
    }
    return EmailText(**{**defaults, **kwargs})


def test_parse_reads_category_and_confidence() -> None:
    decision = parse_llm_answer(
        '{"category": "Transaction", "confidence": 0.77}', model="pegasus-lite"
    )
    assert decision.category == "Transaction"
    assert decision.confidence == pytest.approx(0.77)
    assert decision.backend == "llm"
    assert decision.model == "pegasus-lite"


def test_parse_tolerates_code_fences() -> None:
    decision = parse_llm_answer(
        '```json\n{"category": "Updates", "confidence": 0.6}\n```', model="m"
    )
    assert decision.category == "Updates"


def test_parse_clamps_confidence() -> None:
    decision = parse_llm_answer('{"category": "OTP", "confidence": 1.5}', model="m")
    assert decision.confidence == pytest.approx(1.0)


def test_parse_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        parse_llm_answer('{"category": "Spam", "confidence": 1}', model="m")


def test_parse_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="not JSON"):
        parse_llm_answer("Primary", model="m")


class StubPost:
    def __init__(self, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self.content = content

    def __call__(
        self, url: str, payload: dict[str, Any], *, api_key: str, timeout: int
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "api_key": api_key})
        return {"choices": [{"message": {"content": self.content}}]}


def classifier(post: Any) -> LlmClassifier:
    return LlmClassifier(
        api_base="http://proxy:4000/v1",
        api_key="sk-test",
        model="pegasus-lite",
        post=post,
    )


def test_llm_asks_one_chat_completions_call_per_email() -> None:
    post = StubPost('{"category": "Primary", "confidence": 0.9}')
    decision = classifier(post).classify(email())
    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"] == "http://proxy:4000/v1/chat/completions"
    assert call["api_key"] == "sk-test"
    payload = call["payload"]
    assert payload["model"] == "pegasus-lite"
    assert payload["response_format"] == {"type": "json_object"}
    system, user = payload["messages"]
    assert system["role"] == "system"
    assert "OTP" in system["content"] and "Promotional" in system["content"]
    assert user["role"] == "user" and user["content"].startswith("Sender: Amazon")
    assert decision.category == "Primary"
    assert decision.confidence == pytest.approx(0.9)


def test_close_is_safe_without_transport() -> None:
    classifier(StubPost('{"category": "Updates", "confidence": 0.5}')).close()


@pytest.mark.parametrize(
    ("sender", "subject", "body", "expected"),
    [
        ("GitHub", "Use 442331 as your verification code", "Enter 442331 to sign in", "OTP"),
        ("Google", "Code is 8821", "code is 8821", "OTP"),
        ("Amazon", "Your receipt #111", "Thanks for your purchase", "Transaction"),
        ("Delta", "Booking confirmation 8FX2Q", "your itinerary", "Transaction"),
        ("Acme", "50% off everything this week", "shop now", "Promotional"),
        ("Fancy", "Our March newsletter", "read this", "Updates"),
        ("Ada", "Lunch Friday?", "are you free", "Primary"),
        ("Notifapp", "Bo commented on your post", "unsubscribe here", "Updates"),
    ],
)
def test_rules_backend_happy_path(sender: str, subject: str, body: str, expected: str) -> None:
    rules = RulesClassifier()
    decision = rules.classify(email(sender=sender, subject=subject, body=body))
    assert decision.category == expected
    assert decision.backend == "rules"
    assert decision.confidence == pytest.approx(0.4)
