"""Classifier backends: TypeSafe Jev (System One), OpenAI-compatible LLM, offline keyword rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .categories import (
    CATEGORIES,
    CATEGORY_CRITERIA,
    CLASSIFY_INSTRUCTIONS,
    OTP,
    OTP_NOUL_INSTRUCTIONS,
)
from .config import Config
from .state import EmailText

CATEGORY_QUESTION = "category"
OTP_QUESTION = "delivers_one_time_code"


@dataclass(frozen=True)
class Decision:
    category: str
    confidence: float
    category_probability: float
    otp_signal: float
    backend: str
    model: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "probability": round(self.category_probability, 3),
            "otp_signal": round(self.otp_signal, 3),
            "backend": self.backend,
            "model": self.model,
            "note": self.note,
        }


@runtime_checkable
class Classifier(Protocol):
    name: str

    def classify(self, email: EmailText) -> Decision: ...

    def close(self) -> None: ...


def build_questions(Choice: type, Noul: type) -> dict[str, Any]:
    return {
        CATEGORY_QUESTION: Choice(
            instructions=CLASSIFY_INSTRUCTIONS,
            criteria=dict(CATEGORY_CRITERIA),
        ),
        OTP_QUESTION: Noul(instructions=OTP_NOUL_INSTRUCTIONS),
    }


def decision_from_response(
    response: Any,
    *,
    backend: str,
    model: str,
    otp_override: float,
) -> Decision:
    answers: dict[str, Any] = dict(response.answers)
    choice = answers.get(CATEGORY_QUESTION)
    if choice is None:
        raise ValueError(f"backend returned no {CATEGORY_QUESTION!r} answer")

    category = str(choice.choice)
    if category not in CATEGORIES:
        raise ValueError(f"backend returned unknown category {category!r}")

    probabilities = {
        str(k): float(v) for k, v in (getattr(choice, "probabilities", None) or {}).items()
    }
    noul = answers.get(OTP_QUESTION)
    otp_signal = float(getattr(noul, "noul", 0.0) or 0.0) if noul is not None else 0.0

    note = ""
    if category != OTP and otp_signal >= otp_override:
        note = f"noul override: p(one-time code)={otp_signal:.2f} >= {otp_override:.2f}"
        category = OTP

    return Decision(
        category=category,
        confidence=float(getattr(choice, "confidence", 0.0) or 0.0),
        category_probability=probabilities.get(category, 0.0),
        otp_signal=otp_signal,
        backend=backend,
        model=model,
        note=note,
    )


class SystemOneClassifier:
    """Shared driver for the typesafe-sdk client and the system-one-adapter client."""

    def __init__(
        self,
        *,
        name: str,
        client: Any,
        model: Any,
        model_label: str,
        question_types: tuple[type, type],
        structured_state: bool,
        max_body_chars: int = 1500,
        otp_override: float = 0.9,
    ) -> None:
        self.name = name
        self._client = client
        self._model = model
        self.model_label = model_label
        self._structured_state = structured_state
        self._max_body_chars = max_body_chars
        self._otp_override = otp_override
        self._questions = build_questions(*question_types)

    def classify(self, email: EmailText) -> Decision:
        state: Any = (
            email.state_dict(self._max_body_chars)
            if self._structured_state
            else email.state_text(self._max_body_chars)
        )
        response = self._client.system_one(
            state=state, questions=self._questions, model=self._model
        )
        return decision_from_response(
            response,
            backend=self.name,
            model=self.model_label,
            otp_override=self._otp_override,
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class RulesClassifier:
    """Key-free heuristic backend: useful to rehearse the pipeline without API keys."""

    name = "rules"
    model_label = "keyword-heuristics"

    PATTERNS: tuple[tuple[str, str], ...] = (
        (OTP, r"\b(otp|one[- ]?time (passcode|code)|verification code|security code|passcode)\b"),
        (OTP, r"\bcode (is|:)\s*[A-Z0-9]{4,8}\b"),
        (OTP, r"(magic sign[- ]?in link|confirm your (sign[- ]?in|login))"),
        (
            "Transaction",
            r"\b(receipt|invoice|order (confirmation|update|shipped)|shipping|delivered)",
        ),
        ("Transaction", r"\b(payment (confirmed|received|failed)|booking confirmation|itinerary)"),
        ("Transaction", r"\b(security alert|password (changed|reset)|refund (processed|issued))"),
        (
            "Promotional",
            r"(\d{1,3}%\s*off|\bdiscount\b|\bcoupon\b|limited time|flash sale|\bsale\b)",
        ),
        ("Promotional", r"(upgrade now|start your free trial|webinar|we miss you|come back)"),
        ("Updates", r"\b(newsletter|digest|announcement|release notes|what'?s new)\b"),
        (
            "Updates",
            r"(commented on|shared a (file|document) with you|mentioned you|activity summary)",
        ),
    )

    COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
        (category, re.compile(pattern, re.IGNORECASE)) for category, pattern in PATTERNS
    )

    def classify(self, email: EmailText) -> Decision:
        haystack = f"{email.sender}\n{email.subject}\n{email.excerpt(600)}"
        category = ""
        for candidate, pattern in self.COMPILED:
            if pattern.search(haystack):
                category = candidate
                break
        if not category:
            bulk = re.search(
                r"(unsubscribe|manage preferences|notification settings)", haystack, re.IGNORECASE
            )
            category = "Updates" if bulk else "Primary"
        return Decision(
            category=category,
            confidence=0.4,
            category_probability=0.4,
            otp_signal=1.0 if category == OTP else 0.0,
            backend=self.name,
            model=self.model_label,
            note="heuristic backend - not a real model",
        )

    def close(self) -> None:
        return None


def _jev_classifier(config: Config, *, max_body_chars: int, otp_override: float) -> Classifier:
    from typesafe_sdk import Choice, Noul, TypeSafeClient

    client = TypeSafeClient(api_key=config.typesafe_api_key, model=config.typesafe_model)
    return SystemOneClassifier(
        name="jev",
        client=client,
        model=config.typesafe_model,
        model_label=config.typesafe_model,
        question_types=(Choice, Noul),
        structured_state=True,
        max_body_chars=max_body_chars,
        otp_override=otp_override,
    )


def _llm_classifier(config: Config, *, max_body_chars: int, otp_override: float) -> Classifier:
    from system_one_adapter import Choice, Noul, SystemOneAdapterClient
    from system_one_adapter.providers.openai import OpenAIProvider

    client = SystemOneAdapterClient(
        structured_outputs=True,
        llm_answer_mode="probabilities",
        normalize_probabilities=True,
        n_retry_malformed_structure=2,
    )
    provider = OpenAIProvider(
        config.llm_model,
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
        api="chat_completions",
    )
    return SystemOneClassifier(
        name="llm",
        client=client,
        model=provider,
        model_label=f"{config.llm_model}@{config.llm_api_base}",
        question_types=(Choice, Noul),
        structured_state=False,
        max_body_chars=max_body_chars,
        otp_override=otp_override,
    )


def make_classifier(
    config: Config, *, max_body_chars: int = 1500, otp_override: float = 0.9
) -> Classifier:
    if config.backend == "jev":
        return _jev_classifier(config, max_body_chars=max_body_chars, otp_override=otp_override)
    if config.backend == "llm":
        return _llm_classifier(config, max_body_chars=max_body_chars, otp_override=otp_override)
    return RulesClassifier()
