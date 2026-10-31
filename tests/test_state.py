from __future__ import annotations

from gmail_manager.state import EmailText, clean_text, drop_quoted


def email(**kwargs: object) -> EmailText:
    defaults: dict[str, object] = {
        "message_id": "m1",
        "sender": "Ada <ada@example.com>",
        "recipients": "me@example.com",
        "subject": "Lunch Friday?",
        "date": "Sat, 3 Oct 2026",
        "snippet": "Are we still on for",
        "body": "Hey! Are we still on for lunch Friday?",
    }
    return EmailText(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_clean_text_strips_html_and_collapses_whitespace() -> None:
    raw = "<div><style>body{}</style><h1>Big&nbsp;deal</h1><br>second  line</div>"
    assert clean_text(raw, 500) == "Big deal\nsecond line"


def test_clean_text_truncates() -> None:
    assert clean_text("x" * 50, 10).endswith("…")
    assert len(clean_text("x" * 50, 10)) == 11


def test_drop_quoted_cuts_reply_chain() -> None:
    body = "Thanks, that works.\n\nOn Fri, Sep 30, Lee wrote:\n> long ago\n> more"
    assert drop_quoted(body) == "Thanks, that works."


def test_body_wins_over_snippet() -> None:
    assert "lunch Friday" in email().excerpt(500)


def test_snippet_used_when_no_body() -> None:
    assert email(body="").excerpt(500) == "Are we still on for"


def test_state_text_has_all_context() -> None:
    state = email().state_text(500)
    for fragment in ("Ada <ada@example.com>", "Lunch Friday?", "still on for lunch"):
        assert fragment in state


def test_state_dict_shape() -> None:
    state = email().state_dict(500)
    assert state["sender"] == "Ada <ada@example.com>"
    assert state["subject"] == "Lunch Friday?"
    assert "still on for lunch" in state["message_text"]


def test_missing_subject_becomes_placeholder() -> None:
    assert "(no subject)" in email(subject="").state_text(500)
