"""Command line interface."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import click

from .categories import CATEGORIES
from .classifier import Classifier, Decision, make_classifier
from .config import BACKENDS, Config
from .gmail import (
    GmailAuthError,
    apply_label,
    fetch_email,
    get_service,
    list_labels,
    list_messages,
    resolve_label_ids,
)
from .state import EmailText

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

CATEGORY_COLORS = {
    "OTP": "cyan",
    "Primary": "white",
    "Transaction": "green",
    "Updates": "blue",
    "Promotional": "magenta",
}


@dataclass
class Result:
    email: EmailText
    decision: Decision | None = None
    outcome: str = "pending"
    detail: str = ""
    error: str = ""
    tags: list[str] = field(default_factory=list)


def _parse_label_overrides(pairs: tuple[str, ...]) -> dict[str, str]:
    wanted = {category: category for category in CATEGORIES}
    lookup = {category.lower(): category for category in CATEGORIES}
    for pair in pairs:
        category, separator, label_name = pair.partition("=")
        key = lookup.get(category.strip().lower())
        if not separator or key is None or not label_name.strip():
            raise click.BadParameter(
                f"--label expects CATEGORY=LabelName with CATEGORY one of {', '.join(CATEGORIES)}",
                param_hint="--label",
            )
        wanted[key] = label_name.strip()
    return wanted


def _echo_result(result: Result) -> None:
    email = result.email
    if result.decision is None:
        click.secho(
            f"  ERROR  {email.sender[:28]:<28} {email.headline[:46]:<46} {result.error}", fg="red"
        )
        return
    decision = result.decision
    color = CATEGORY_COLORS.get(decision.category, "white")
    extra = " ".join(result.tags + ([result.detail] if result.detail else []))
    click.echo(
        click.style(f"  {decision.category.ljust(12)}", fg=color, bold=True)
        + f" p={decision.category_probability:.2f} c={decision.confidence:.2f}"
        + f"  {email.sender[:26]:<26} {email.headline[:44]:<44}"
        + click.style(f" {extra}", dim=True)
    )
    if decision.note:
        click.echo(click.style(f"              note: {decision.note}", dim=True))
    if result.error:
        click.secho(f"              error: {result.error}", fg="red")


def _classify_all(
    classifier: Classifier, emails: Iterable[EmailText], workers: int
) -> list[Result]:
    items = list(emails)

    def run(email: EmailText) -> Result:
        try:
            return Result(email=email, decision=classifier.classify(email))
        except Exception as exc:
            return Result(email=email, outcome="error", error=f"{type(exc).__name__}: {exc}")

    if workers <= 1 or len(items) <= 1:
        return [run(email) for email in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, items))


def _finalize(
    result: Result,
    *,
    service: Any,
    label_ids: dict[str, str],
    min_confidence: float,
    user_id: str,
    dry_run: bool,
) -> Result:
    decision = result.decision
    if decision is None:
        return result
    label_name = label_ids.get(decision.category)
    if decision.confidence < min_confidence:
        result.outcome = "low-confidence"
        result.tags.append(f"conf<{min_confidence:g}")
        return result
    if label_name is None:
        result.outcome = "no-label"
        result.tags.append("no matching Gmail label")
        return result
    if label_name in result.email.label_ids:
        result.outcome = "already-labelled"
        result.tags.append("label already present")
        return result
    if dry_run:
        result.outcome = "would-label"
        result.detail = f"would apply {label_name}"
        return result
    try:
        apply_label(service, result.email.message_id, label_name, user_id=user_id)
    except Exception as exc:
        result.outcome = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.outcome = "labelled"
    result.detail = f"applied {label_name}"
    return result


@click.group(context_settings=CONTEXT_SETTINGS)
def main() -> None:
    """Classify Gmail inbox mail into five categories and apply existing Gmail labels."""


@main.command()
@click.option(
    "--force", is_flag=True, help="Redo the OAuth consent screen even with a cached token."
)
@click.option("--check", is_flag=True, help="Only test the cached token and print the account.")
def auth(force: bool, check: bool) -> None:
    """Authorize Gmail access (writes token.json)."""
    config = Config.from_env()
    if check and not config.token_file.exists():
        raise click.ClickException(
            f"no cached token at {config.token_file}; run `gmail-manager auth` first"
        )
    try:
        service = get_service(config.credentials_file, config.token_file, force_auth=force)
    except GmailAuthError as exc:
        raise click.ClickException(str(exc)) from None
    profile = service.users().getProfile(userId=config.user_id).execute()
    click.secho(f"authorized as {profile.get('emailAddress', config.user_id)}", fg="green")
    click.echo(f"token: {config.token_file}")


@main.command()
@click.option(
    "--label", "label_overrides", multiple=True, help="Override the label for a category."
)
def labels(label_overrides: tuple[str, ...]) -> None:
    """Show which existing Gmail labels the five categories map to."""
    config = Config.from_env()
    try:
        service = get_service(config.credentials_file, config.token_file)
    except GmailAuthError as exc:
        raise click.ClickException(str(exc)) from None

    wanted = _parse_label_overrides(label_overrides)
    label_ids, missing = resolve_label_ids(service, wanted)
    by_id = {label["id"]: label for label in list_labels(service)}
    click.echo(f"{len(by_id)} labels in this mailbox\n")
    for category in CATEGORIES:
        name = wanted[category]
        if category in label_ids:
            click.secho(
                f"  {category.ljust(12)} -> {by_id[label_ids[category]]['name']}", fg="green"
            )
        else:
            click.secho(f"  {category.ljust(12)} -> no label named '{name}'", fg="yellow")
    if missing:
        click.echo(
            "\ncreate the missing labels in Gmail, or remap them with --label CATEGORY=LabelName"
        )
    used = set(label_ids.values())
    others = [
        label for label in by_id.values() if label["type"] == "user" and label["id"] not in used
    ]
    if others:
        click.echo("\nother user labels:")
        for label in sorted(others, key=lambda item: item["name"].lower()):
            click.echo(f"  {label['name'][:44]}")


@main.command()
@click.option(
    "-q", "--query", default="in:inbox", show_default=True, help="Any Gmail search query."
)
@click.option(
    "-n", "--max", "max_results", default=50, show_default=True, help="Max messages to process."
)
@click.option(
    "--backend",
    type=click.Choice(BACKENDS),
    default=None,
    help="Default: jev, or llm without a key.",
)
@click.option(
    "--confidence", default=0.55, show_default=True, help="Skip labelling below this confidence."
)
@click.option("--workers", default=4, show_default=True, help="Concurrent classifier calls.")
@click.option(
    "--body-chars", default=1500, show_default=True, help="Body characters sent to the model."
)
@click.option(
    "--otp-override", default=0.9, show_default=True, help="Noul p(one-time code) that forces OTP."
)
@click.option("--no-bodies", is_flag=True, help="Classify from headers and snippet only (cheaper).")
@click.option(
    "--reclassify", is_flag=True, help="Re-run on messages that already carry a managed label."
)
@click.option("--dry-run", is_flag=True, help="Classify without touching the mailbox.")
@click.option(
    "--label", "label_overrides", multiple=True, help="Override the label for a category."
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON lines instead of a table.")
def classify(
    query: str,
    max_results: int,
    backend: str | None,
    confidence: float,
    workers: int,
    body_chars: int,
    otp_override: float,
    no_bodies: bool,
    reclassify: bool,
    dry_run: bool,
    label_overrides: tuple[str, ...],
    as_json: bool,
) -> None:
    """Classify messages into the five inbox categories, then apply the matching Gmail label."""
    try:
        config = Config.from_env(backend)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    for note in config.notes:
        click.secho(f"warning: {note}", fg="yellow", err=True)

    wanted = _parse_label_overrides(label_overrides)
    try:
        service = get_service(config.credentials_file, config.token_file)
    except GmailAuthError as exc:
        raise click.ClickException(str(exc)) from None

    label_ids, missing = resolve_label_ids(service, wanted)
    for item in missing:
        click.secho(f"warning: no Gmail label for {item} - reporting only", fg="yellow", err=True)
    managed = set(label_ids.values())

    mode = "dry-run" if dry_run else "apply"
    click.echo(f"backend={config.backend}  query={query!r}  max={max_results}  mode={mode}")

    emails: list[EmailText] = []
    skipped = 0
    for message in list_messages(
        service, query=query, user_id=config.user_id, max_results=max_results
    ):
        email = fetch_email(service, message, user_id=config.user_id, include_body=not no_bodies)
        if not reclassify and managed & set(email.label_ids):
            skipped += 1
            continue
        emails.append(email)

    if not emails:
        click.echo(f"nothing to classify (skipped {skipped} already labelled)")
        return

    click.echo(f"classifying {len(emails)} messages with {config.backend}…\n")
    classifier = make_classifier(config, max_body_chars=body_chars, otp_override=otp_override)
    try:
        results = _classify_all(classifier, emails, workers)
    finally:
        classifier.close()

    for result in results:
        _finalize(
            result,
            service=service,
            label_ids=label_ids,
            min_confidence=confidence,
            user_id=config.user_id,
            dry_run=dry_run,
        )
        if as_json:
            click.echo(json.dumps(_result_dict(result, wanted)))
        else:
            _echo_result(result)

    counts = Counter(result.outcome for result in results)
    by_category = Counter(r.decision.category for r in results if r.decision is not None)
    click.echo(
        "\n"
        + "  ".join(
            f"{category}={by_category[category]}"
            for category in CATEGORIES
            if by_category[category]
        )
    )
    outcomes = [f"{key}={value}" for key, value in sorted(counts.items()) if key != "labelled"]
    click.echo(
        f"{len(results)} messages: "
        + "  ".join([f"labelled={counts['labelled']}", *outcomes])
        + (f"  skipped(already labelled)={skipped}" if skipped else "")
    )
    if counts["error"] or counts["failed"]:
        raise SystemExit(1)


def _result_dict(result: Result, wanted: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messageId": result.email.message_id,
        "threadId": result.email.thread_id,
        "sender": result.email.sender,
        "subject": result.email.subject,
        "outcome": result.outcome,
        "detail": result.detail or " ".join(result.tags),
        "label": wanted.get(result.decision.category) if result.decision else None,
    }
    if result.decision is not None:
        payload["decision"] = result.decision.to_dict()
    if result.error:
        payload["error"] = result.error
    return payload


if __name__ == "__main__":
    main()
