"""Gmail plumbing: OAuth, inbox reads, label lookup and label writes."""

from __future__ import annotations

import base64
import quopri
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from .state import EmailText

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
HEADER_NAMES = ("From", "To", "Cc", "Subject", "Date")
OAUTH_SUCCESS_HTML = "Authorization complete. You can close this tab and return to the terminal."


class GmailAuthError(RuntimeError):
    pass


class GmailAPIError(RuntimeError):
    pass


def _write_token(token_file: Path, credentials: Credentials) -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    token_file.chmod(0o600)


def authorize(credentials_file: Path, token_file: Path, *, force: bool = False) -> Credentials:
    credentials: Credentials | None = None
    if token_file.exists() and not force:
        credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    can_refresh = (
        credentials is not None
        and credentials.expired
        and not credentials.valid
        and bool(credentials.refresh_token)
    )
    if can_refresh and credentials is not None:
        try:
            credentials.refresh(Request())
        except RefreshError:
            credentials = None
        else:
            _write_token(token_file, credentials)

    if credentials is None or not credentials.valid:
        if not credentials_file.exists():
            raise GmailAuthError(
                f"no OAuth client found at {credentials_file}. Create an OAuth 2.0 'Desktop app' "
                "client in the Google Cloud console (with the Gmail API enabled), download its "
                "JSON there, then rerun `gmail-manager auth`."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
        credentials = flow.run_local_server(
            port=0, prompt="consent", success_message=OAUTH_SUCCESS_HTML
        )
        _write_token(token_file, credentials)

    return credentials


def get_service(credentials_file: Path, token_file: Path, *, force_auth: bool = False) -> Resource:
    credentials = authorize(credentials_file, token_file, force=force_auth)
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _decode_part(part: Mapping[str, Any]) -> str:
    data = (part.get("body") or {}).get("data")
    if not data:
        return ""
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    headers = {h.get("name", "").lower(): h.get("value", "") for h in part.get("headers") or []}
    if "quoted-printable" in headers.get("content-transfer-encoding", "").lower():
        raw = quopri.decodestring(raw)
    return raw.decode("utf-8", errors="replace")


def extract_body(payload: Mapping[str, Any], max_bytes: int = 20000) -> str:
    plain: list[str] = []
    html: list[str] = []
    total = 0
    stack: list[Mapping[str, Any]] = [payload]
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        if part.get("filename"):
            continue
        data = (part.get("body") or {}).get("data")
        if data and mime in {"text/plain", "text/html"} and total < max_bytes:
            text = _decode_part(part)
            total += len(text)
            (plain if mime == "text/plain" else html).append(text)
        stack.extend(part.get("parts") or [])
    body = "\n\n".join(plain).strip() or "\n\n".join(html).strip()
    return body[:max_bytes]


def header_values(payload: Mapping[str, Any]) -> dict[str, str]:
    headers = payload.get("headers") or []
    out: dict[str, str] = {}
    for header in headers:
        name = header.get("name", "")
        if name in HEADER_NAMES and name not in out:
            out[name] = header.get("value", "")
    return out


def list_messages(
    service: Resource,
    *,
    query: str,
    user_id: str = "me",
    max_results: int = 50,
) -> Iterator[dict[str, Any]]:
    page_token: str | None = None
    fetched = 0
    while fetched < max_results:
        request = (
            service.users()
            .messages()
            .list(
                userId=user_id,
                q=query,
                maxResults=min(250, max_results - fetched),
                pageToken=page_token,
            )
        )
        response = request.execute()
        messages = response.get("messages") or []
        for message in messages:
            yield message
            fetched += 1
            if fetched >= max_results:
                return
        page_token = response.get("nextPageToken")
        if not page_token or not messages:
            return


def fetch_email(
    service: Resource,
    message: Mapping[str, Any],
    *,
    user_id: str = "me",
    include_body: bool = True,
) -> EmailText:
    message_id = message["id"]
    if include_body:
        raw = service.users().messages().get(userId=user_id, id=message_id, format="full").execute()
    else:
        raw = (
            service.users()
            .messages()
            .get(
                userId=user_id,
                id=message_id,
                format="metadata",
                metadataHeaders=list(HEADER_NAMES),
            )
            .execute()
        )
    headers = header_values(raw.get("payload") or {})
    body = extract_body(raw["payload"]) if (include_body and raw.get("payload")) else ""
    return EmailText(
        message_id=message_id,
        thread_id=raw.get("threadId", message.get("threadId", "")),
        subject=headers.get("Subject", ""),
        sender=headers.get("From", ""),
        recipients=", ".join(v for v in (headers.get("To"), headers.get("Cc")) if v),
        date=headers.get("Date", ""),
        snippet=raw.get("snippet", message.get("snippet", "")),
        body=body,
        label_ids=tuple(raw.get("labelIds") or message.get("labelIds") or ()),
    )


def list_labels(service: Resource) -> list[dict[str, Any]]:
    return list(service.users().labels().list(userId="me").execute().get("labels", []))


def resolve_label_ids(
    service: Resource,
    wanted: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Map category -> Gmail label id for labels that already exist. Returns (resolved, missing)."""
    existing = {label["name"].strip().lower(): label["id"] for label in list_labels(service)}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for category, label_name in wanted.items():
        label_id = existing.get(label_name.strip().lower())
        if label_id:
            resolved[category] = label_id
        else:
            missing.append(f"{category} -> '{label_name}'")
    return resolved, missing


def apply_label(service: Resource, message_id: str, label_id: str, *, user_id: str = "me") -> None:
    service.users().messages().modify(
        userId=user_id,
        id=message_id,
        body={"addLabelIds": [label_id]},
    ).execute()
