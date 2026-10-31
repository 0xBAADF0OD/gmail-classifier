from __future__ import annotations

from typing import Any

import pytest

from gmail_manager.categories import CATEGORIES
from gmail_manager.classifier import (
    CATEGORY_QUESTION,
    OTP_QUESTION,
    RulesClassifier,
    SystemOneClassifier,
    decision_from_response,
)
from gmail_manager.state import EmailText


class Answer:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class Response:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers


def choice(category: str, confidence: float = 0.8, **probabilities: float) -> Answer:
    probs = probabilities or {name: (0.9 if name == category else 0.02) for name in CATEGORIES}
    return Answer(type="choice", choice=category, confidence=confidence, probabilities=probs)


def email(**kwargs: Any) -> EmailText:
    defaults: dict[str, Any] = {
        "message_id": "m1",
        "sender": "Amazon <auto-confirm@amazon.com>",
        "subject": "Your order has shipped",
        "body": "Your order #112 has shipped and arrives Tuesday.",
    }
    return EmailText(**{**defaults, **kwargs})


def test_decision_reads_choice_and_noul() -> None:
    response = Response(
        {
            CATEGORY_QUESTION: choice("Transaction", 0.77, Transaction=0.8),
            OTP_QUESTION: Answer(noul=0.1),
        }
    )
    decision = decision_from_response(response, backend="jev", model="jev-latest", otp_override=0.9)
    assert decision.category == "Transaction"
    assert decision.confidence == pytest.approx(0.77)
    assert decision.category_probability == pytest.approx(0.8)
    assert decision.note == ""


def test_noul_overrides_a_non_otp_answer() -> None:
    response = Response(
        {
            CATEGORY_QUESTION: choice("Transaction", 0.8, Transaction=0.8, OTP=0.1),
            OTP_QUESTION: Answer(noul=0.95),
        }
    )
    decision = decision_from_response(response, backend="jev", model="jev", otp_override=0.9)
    assert decision.category == "OTP"
    assert "noul override" in decision.note


def test_noul_below_threshold_leaves_answer_alone() -> None:
    response = Response(
        {CATEGORY_QUESTION: choice("Updates", 0.8, Updates=0.8), OTP_QUESTION: Answer(noul=0.4)}
    )
    decision = decision_from_response(response, backend="jev", model="jev", otp_override=0.9)
    assert decision.category == "Updates"


def test_missing_answer_raises() -> None:
    with pytest.raises(ValueError, match="category"):
        decision_from_response(Response({}), backend="jev", model="jev", otp_override=0.9)


def test_unknown_category_raises() -> None:
    response = Response(
        {CATEGORY_QUESTION: Answer(choice="Spam", confidence=1.0, probabilities={})}
    )
    with pytest.raises(ValueError, match="unknown category"):
        decision_from_response(response, backend="jev", model="jev", otp_override=0.9)


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def system_one(self, **kwargs: Any) -> Response:
        self.calls.append(kwargs)
        return Response(
            {
                CATEGORY_QUESTION: choice("Primary", 0.9, Primary=0.9),
                OTP_QUESTION: Answer(noul=0.0),
            }
        )


class FakeChoice:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeNoul:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def classifier(client: Any, structured: bool) -> SystemOneClassifier:
    return SystemOneClassifier(
        name="fake",
        client=client,
        model="fake-model",
        model_label="fake-model",
        question_types=(FakeChoice, FakeNoul),
        structured_state=structured,
    )


def test_jev_style_state_is_a_dict_and_asks_two_questions() -> None:
    client = RecordingClient()
    classifier(client, structured=True).classify(email())
    call = client.calls[0]
    assert call["model"] == "fake-model"
    assert isinstance(call["state"], dict)
    assert call["state"]["subject"] == "Your order has shipped"
    assert set(call["questions"]) == {CATEGORY_QUESTION, OTP_QUESTION}
    assert set(call["questions"][CATEGORY_QUESTION].kwargs["criteria"]) == set(CATEGORIES)


def test_llm_style_state_is_readable_text() -> None:
    client = RecordingClient()
    classifier(client, structured=False).classify(email())
    state = client.calls[0]["state"]
    assert isinstance(state, str)
    assert state.startswith("Sender: Amazon")


def test_close_is_forwarded_when_available() -> None:
    closed: list[str] = []

    class Closable(RecordingClient):
        def close(self) -> None:
            closed.append("closed")

    classifier(Closable(), structured=True).close()
    assert closed == ["closed"]


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
