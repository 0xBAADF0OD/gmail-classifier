"""Normalized email content plus the two state shapes the models are queried with."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

QUOTE_MARKERS = (
    "\n>",
    "\nOn ",
    "\n-----Original Message-----",
    "\n-- \n",
)

STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
BLOCK_END_RE = re.compile(r"</(p|div|tr|li|h[1-6])>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RE = re.compile(r"\n{2,}")


def strip_html(markup: str) -> str:
    text = STYLE_RE.sub(" ", markup)
    text = BREAK_RE.sub("\n", text)
    text = BLOCK_END_RE.sub("\n", text)
    return html.unescape(TAG_RE.sub(" ", text))


def drop_quoted(body: str) -> str:
    cut = len(body)
    for marker in QUOTE_MARKERS:
        index = body.find(marker)
        if index != -1:
            cut = min(cut, index)
    return body[:cut].strip() or body.strip()


def clean_text(text: str, max_chars: int) -> str:
    markup = "<" in text and ">" in text
    text = strip_html(text) if markup else html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = SPACE_RE.sub(" ", text)
    text = NEWLINE_RE.sub("\n", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "…"
    return text


@dataclass(frozen=True)
class EmailText:
    message_id: str
    subject: str = ""
    sender: str = ""
    recipients: str = ""
    date: str = ""
    snippet: str = ""
    body: str = ""
    label_ids: tuple[str, ...] = ()
    thread_id: str = ""

    @property
    def headline(self) -> str:
        return self.subject or "(no subject)"

    def excerpt(self, max_body_chars: int, max_snippet_chars: int = 400) -> str:
        if self.body:
            return clean_text(drop_quoted(self.body), max_body_chars)
        return clean_text(self.snippet, max_snippet_chars)

    def state_text(self, max_body_chars: int = 1500) -> str:
        return "\n".join(
            [
                f"Sender: {self.sender or 'unknown'}",
                f"Recipients: {self.recipients or 'unknown'}",
                f"Subject: {self.headline}",
                f"Date: {self.date or 'unknown'}",
                "",
                "Message text:",
                self.excerpt(max_body_chars) or "(no readable text extracted)",
            ]
        )

    def state_dict(self, max_body_chars: int = 1500) -> dict[str, str]:
        return {
            "sender": self.sender or "unknown",
            "recipients": self.recipients or "unknown",
            "subject": self.headline,
            "date": self.date or "unknown",
            "message_text": self.excerpt(max_body_chars) or "(no readable text extracted)",
        }
