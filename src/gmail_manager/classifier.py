"""Classifier backends: chat-completions against any OpenAI-compatible LLM, plus offline rules."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .categories import CATEGORY_CRITERIA, OTP
from .config import Config
from .state import EmailText

SYSTEM_PROMPT = (
    "Sort the email into exactly one inbox category.\n"
    + "\n".join(f"- {name}: {text}" for name, text in CATEGORY_CRITERIA.items())
    + "\nJudge the whole message. Reply with this JSON object only:\n"
    '{"category": "<one of the five names>", "confidence": <0.0 to 1.0>}\n'
    "Lower the confidence when the message fits several categories."
)


@dataclass(frozen=True)
class Decision:
    category: str
    confidence: float
    backend: str
    model: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "backend": self.backend,
            "model": self.model,
            "note": self.note,
        }


@runtime_checkable
class Classifier(Protocol):
    name: str

    def classify(self, email: EmailText) -> Decision: ...

    def close(self) -> None: ...


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def parse_llm_answer(content: str, *, model: str) -> Decision:
    try:
        data = json.loads(_strip_fences(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"llm answer is not JSON: {exc}") from exc
    category = str(data.get("category", ""))
    if category not in CATEGORY_CRITERIA:
        raise ValueError(f"unknown category {category!r}")
    confidence = float(data.get("confidence", 0.0))
    return Decision(
        category=category,
        confidence=min(max(confidence, 0.0), 1.0),
        backend="llm",
        model=model,
    )


def http_post(url: str, payload: dict[str, Any], *, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return body


class LlmClassifier:
    """One chat-completions call per email; the reply is category + confidence JSON."""

    name = "llm"

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        max_body_chars: int = 1500,
        timeout: int = 30,
        post: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.model = model
        self._max_body_chars = max_body_chars
        self._timeout = timeout
        self._post = post or http_post

    def classify(self, email: EmailText) -> Decision:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": email.state_text(self._max_body_chars)},
            ],
        }
        result = self._post(
            f"{self.api_base}/chat/completions",
            payload,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        content = str(result["choices"][0]["message"]["content"])
        return parse_llm_answer(content, model=self.model)

    def close(self) -> None:
        return None


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
            backend=self.name,
            model=self.model_label,
            note="heuristic backend - not a real model",
        )

    def close(self) -> None:
        return None


def make_classifier(config: Config, *, max_body_chars: int = 1500) -> Classifier:
    if config.backend == "llm":
        return LlmClassifier(
            api_base=config.llm_api_base,
            api_key=config.llm_api_key,
            model=config.llm_model,
            max_body_chars=max_body_chars,
        )
    return RulesClassifier()
