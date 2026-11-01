from __future__ import annotations

from typing import Any

from conftest import FakeGmail, b64, default_labels, make_payload, raw_message
from gmail_manager import gmail
from gmail_manager.categories import DEFAULT_LABEL_NAMES


def test_extract_body_prefers_plain_over_html() -> None:
    payload = make_payload(
        {"From": "a@b.c"},
        [
            {
                "mimeType": "text/html",
                "body": {"data": b64("<p>html body</p>")},
            },
            {
                "mimeType": "text/plain",
                "body": {"data": b64("plain body")},
            },
        ],
    )
    assert gmail.extract_body(payload) == "plain body"


def test_extract_body_falls_back_to_html() -> None:
    payload = make_payload(
        {}, [{"mimeType": "text/html", "body": {"data": b64("<p>hi <b>there</b></p>")}}]
    )
    assert "hi" in gmail.extract_body(payload) and "there" in gmail.extract_body(payload)


def test_extract_body_walks_nested_multipart_and_skips_attachments() -> None:
    payload = make_payload(
        {},
        [
            {
                "mimeType": "multipart/mixed",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": b64("inner text")}},
                    {
                        "mimeType": "application/pdf",
                        "filename": "ticket.pdf",
                        "body": {"data": b64("binary-ish stuff")},
                    },
                ],
            }
        ],
    )
    assert gmail.extract_body(payload) == "inner text"


def test_extract_body_decodes_quoted_printable() -> None:
    payload = make_payload(
        {},
        [
            {
                "mimeType": "text/plain",
                "headers": [{"name": "Content-Transfer-Encoding", "value": "quoted-printable"}],
                "body": {"data": b64("caf=C3=A9 receipt")},
            }
        ],
    )
    assert gmail.extract_body(payload) == "café receipt"


def test_header_values_keeps_only_wanted_headers() -> None:
    payload = make_payload(
        {"From": "a@b.c", "Subject": "Hi", "Date": "today", "Message-ID": "<x>"},
        [],
    )
    headers = gmail.header_values(payload)
    assert headers == {"From": "a@b.c", "Subject": "Hi", "Date": "today"}


def test_list_messages_paginates_and_respects_max() -> None:
    pages = [
        {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "p2"},
        {"messages": [{"id": "c"}, {"id": "d"}]},
    ]
    service = FakeGmail(pages=pages)
    ids = [
        message["id"] for message in gmail.list_messages(service, query="in:inbox", max_results=3)
    ]
    assert ids == ["a", "b", "c"]
    assert service.list_calls[1]["pageToken"] == "p2"


def test_list_messages_unlimited_drains_all_pages() -> None:
    pages = [
        {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "p2"},
        {"messages": [{"id": "c"}]},
    ]
    service = FakeGmail(pages=pages)
    ids = [message["id"] for message in gmail.list_messages(service, query="in:inbox")]
    assert ids == ["a", "b", "c"]
    assert len(service.list_calls) == 2


def test_list_messages_stops_when_pages_cycle() -> None:
    service = FakeGmail(pages=[{"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "t"}])
    ids = [message["id"] for message in gmail.list_messages(service, query="in:inbox")]
    assert ids == ["a", "b"]
    assert len(service.list_calls) == gmail.MAX_REPEATS + 1


def test_fetch_email_reads_headers_body_and_labels() -> None:
    raw = raw_message(
        "m1",
        sender="GitHub <noreply@github.com>",
        subject="Your one-time code is 442331",
        body="Use 442331 to sign in. It expires in 10 minutes.",
        label_ids=["INBOX", "UNREAD"],
    )
    service = FakeGmail({"m1": raw})
    email = gmail.fetch_email(service, {"id": "m1"})
    assert email.sender == "GitHub <noreply@github.com>"
    assert "442331" in email.body
    assert email.snippet
    assert set(email.label_ids) == {"INBOX", "UNREAD"}


def test_fetch_email_without_body_uses_metadata_format() -> None:
    raw = raw_message("m2", sender="x@y.z", subject="s", body="secret body")
    service = FakeGmail({"m2": raw})
    email = gmail.fetch_email(service, {"id": "m2"}, include_body=False)
    assert email.body == ""
    assert service.get_calls[0]["format"] == "metadata"


def test_resolve_label_ids_matches_case_insensitively() -> None:
    service = FakeGmail(labels=default_labels())
    resolved, missing = gmail.resolve_label_ids(service, DEFAULT_LABEL_NAMES)
    assert set(resolved) == set(DEFAULT_LABEL_NAMES)
    assert missing == []


def test_resolve_label_ids_reports_missing() -> None:
    service = FakeGmail(labels=[{"id": "LL_OTP", "name": "OTP", "type": "user"}])
    resolved, missing = gmail.resolve_label_ids(
        service,
        {"OTP": "OTP", "Primary": "People"},
    )
    assert resolved == {"OTP": "LL_OTP"}
    assert missing == ["Primary -> 'People'"]


def test_apply_label_calls_modify() -> None:
    service = FakeGmail({"m3": raw_message("m3", sender="a", subject="b", body="c")})
    gmail.apply_label(service, "m3", "LL_OTP")
    call: dict[str, Any] = service.modify_calls[0]
    assert call["id"] == "m3"
    assert call["body"] == {"addLabelIds": ["LL_OTP"], "removeLabelIds": ["INBOX"]}


def test_apply_label_keep_inbox_preserves_inbox() -> None:
    service = FakeGmail({"m4": raw_message("m4", sender="a", subject="b", body="c")})
    gmail.apply_label(service, "m4", "LL_PRIMARY", keep_inbox=True)
    assert service.modify_calls[0]["body"] == {"addLabelIds": ["LL_PRIMARY"]}
