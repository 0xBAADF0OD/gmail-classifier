from __future__ import annotations

import base64
from typing import Any

import pytest

from gmail_manager.categories import DEFAULT_LABEL_NAMES


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def make_payload(headers: dict[str, str], parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"headers": [{"name": k, "value": v} for k, v in headers.items()], "parts": parts}


def raw_message(
    message_id: str,
    *,
    sender: str,
    subject: str,
    body: str,
    snippet: str = "",
    label_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "threadId": f"t_{message_id}",
        "snippet": snippet or body[:80],
        "labelIds": label_ids or ["INBOX", "UNREAD"],
        "payload": make_payload(
            {
                "From": sender,
                "To": "me@example.com",
                "Subject": subject,
                "Date": "Sat, 3 Oct 2026 09:00:00 +0800",
            },
            [
                {
                    "mimeType": "text/plain",
                    "headers": [{"name": "Content-Type", "value": "text/plain; charset=UTF-8"}],
                    "body": {"data": b64(body), "size": len(body)},
                }
            ],
        ),
    }


def default_labels() -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = [
        {"id": "INBOX", "name": "INBOX", "type": "system"},
        {"id": "CATEGORY_PROMOTIONS", "name": "CATEGORY_PROMOTIONS", "type": "system"},
    ]
    labels.extend(
        {"id": f"LL_{category.upper()}", "name": DEFAULT_LABEL_NAMES[category], "type": "user"}
        for category in DEFAULT_LABEL_NAMES
    )
    return labels


class FakeRequest:
    def __init__(self, produce: Any) -> None:
        self._produce = produce

    def execute(self) -> Any:
        return self._produce()


class FakeMessages:
    def __init__(self, gmail: FakeGmail) -> None:
        self._gmail = gmail

    def list(self, **kwargs: Any) -> FakeRequest:
        self._gmail.list_calls.append(kwargs)
        index = min(len(self._gmail.list_calls) - 1, len(self._gmail.pages) - 1)
        page = self._gmail.pages[index]
        return FakeRequest(lambda: page)

    def get(self, **kwargs: Any) -> FakeRequest:
        self._gmail.get_calls.append(kwargs)
        message = self._gmail.messages_by_id[kwargs["id"]]
        return FakeRequest(lambda: message)

    def modify(self, **kwargs: Any) -> FakeRequest:
        self._gmail.modify_calls.append(kwargs)
        return FakeRequest(lambda: dict(self._gmail.messages_by_id[kwargs["id"]]))


class FakeLabels:
    def __init__(self, gmail: FakeGmail) -> None:
        self._gmail = gmail

    def list(self, **kwargs: Any) -> FakeRequest:
        return FakeRequest(lambda: {"labels": self._gmail.labels})


class FakeUsers:
    def __init__(self, gmail: FakeGmail) -> None:
        self._gmail = gmail

    def getProfile(self, userId: str = "me") -> FakeRequest:
        return FakeRequest(lambda: {"emailAddress": "me@example.com"})

    def messages(self) -> FakeMessages:
        return FakeMessages(self._gmail)

    def labels(self) -> FakeLabels:
        return FakeLabels(self._gmail)


class FakeGmail:
    """Duck-typed stand-in for the googleapiclient gmail resource."""

    def __init__(
        self,
        messages_by_id: dict[str, dict[str, Any]] | None = None,
        labels: list[dict[str, Any]] | None = None,
        pages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.messages_by_id = messages_by_id or {}
        self.labels = labels if labels is not None else default_labels()
        self.pages = (
            pages
            if pages is not None
            else [
                {"messages": [{"id": key, "threadId": f"t_{key}"} for key in self.messages_by_id]}
            ]
        )
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.modify_calls: list[dict[str, Any]] = []

    def users(self) -> FakeUsers:
        return FakeUsers(self)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_API_KEY",
        "LLM_API_BASE",
        "LLM_MODEL",
        "CLASSIFIER_BACKEND",
        "GMAIL_MANAGER_DEBUG",
        "GMAIL_USER_ID",
        "GMAIL_CREDENTIALS_FILE",
        "GMAIL_TOKEN_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GMAIL_CREDENTIALS_FILE", "/tmp/gmail-manager-test-client.json")
    monkeypatch.setenv("GMAIL_TOKEN_FILE", "/tmp/gmail-manager-test-token.json")
