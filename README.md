# gmail-manager

A CLI that reads your Gmail inbox, classifies each message into **OTP / Primary / Transaction /
Updates / Promotional** with an LLM, and applies the matching **existing** Gmail label.
It never creates labels and never touches Gmail's own category labels. Messages answered with a
confidence below the threshold go to `Categories/Unsure`.

Two classifier backends, one interface:

| Backend | Engine | Shape of the answer |
| --- | --- | --- |
| `llm` (default) | any OpenAI-compatible chat-completions endpoint (your LiteLLM proxy / `pegasus-lite`) | forced JSON: `{"category": ..., "confidence": ...}` |
| `rules` | keyword heuristics, no API key or network | fixed 0.4 confidence |

Every classification is one API call asking a single question: which of the five categories best
describes the message. The answer is `category + confidence`, nothing else.

## 1. Install

```bash
uv sync --all-groups        # creates .venv on Python 3.12 and installs everything
```

## 2. Gmail credentials

1. In the [Google Cloud console](https://console.cloud.google.com/apis/gmail), enable the **Gmail API**.
2. Configure the OAuth consent screen (External, add your own address as a test user).
3. Create an **OAuth client ID** of type **Desktop app** and download the JSON.
4. Save it as `credentials.json` in this folder (or point `GMAIL_CREDENTIALS_FILE` at it), then:

```bash
uv run gmail-manager auth          # opens a browser, writes token.json (chmod 600)
uv run gmail-manager auth --check  # verify the cached token
```

Scope requested: `https://www.googleapis.com/auth/gmail.modify`.

## 3. Labels

The tool only applies labels that **already exist** in Gmail, matched by name (case-insensitive).
Create them in Gmail first: `Categories/OTP`, `Categories/Primary`, `Categories/Transactions`,
`Categories/Updates`, `Categories/Promotions`, `Categories/Unsure`.

```bash
uv run gmail-manager labels                      # show the mapping and what's missing
uv run gmail-manager labels --label Updates=News # remap a category to another label name
```

Anything with no matching label is classified and reported, never labelled.

## 4. Classify

```bash
cp .env.example .env                            # then fill in the keys
uv run gmail-manager classify --dry-run -n 20    # nothing is written to the mailbox
uv run gmail-manager classify                    # classify + label every fresh inbox message
uv run gmail-manager classify -q "in:inbox is:unread" --confidence 0.8
uv run gmail-manager classify --json > run.jsonl
uv run gmail-manager classify --no-bodies -n 200  # headers + snippet only, cheaper and faster
```

Sample output:

```
backend=llm  query='in:inbox'  max=all  mode=apply
classifying 12 messages with llm…

  OTP          c=0.96  GitHub <noreply@github.c  Your GitHub verification code is… applied Categories/OTP
  Transaction  c=0.91  Amazon <auto-confirm@amz  Your delivery has arrived        applied Categories/Transactions
  Updates      c=0.41  MuseScore <info@mail.mus  Final hours of 90% off MuseScore conf<0.66 applied Categories/Unsure

12 messages: OTP=3 Transaction=4 Promotional=5 low-confidence=3 skipped(already labelled)=6
```

Intake reads messages that carry nothing beyond `INBOX`/`UNREAD`; anything with another label is
treated as already triaged and skipped unless you pass `--reclassify`. Applying a label also removes
`INBOX` so the message leaves the intake queue — except `Categories/Primary`, whose messages stay
in the inbox.

Set `GMAIL_MANAGER_DEBUG=1` to watch progress while a run is in flight: each message prints a
`[i/n] CATEGORY c=0.xx sender subject` line the moment it is classified, and the end-of-run table
is skipped (JSON lines still print with `--json`).

## Backends

`CLASSIFIER_BACKEND` (or `--backend`) picks the engine; the default is `llm`.

**LLM** — any OpenAI-compatible chat-completions endpoint. One call per email: a short system
prompt with the five category definitions, the email as the user message, and `response_format: json_object`
forcing a `{"category", "confidence"}` reply. Point `LLM_API_BASE` at your LiteLLM proxy:

```dotenv
CLASSIFIER_BACKEND=llm
LLM_API_BASE=http://172.16.1.51:4000/v1
LLM_API_KEY=<your litellm key>
LLM_MODEL=pegasus-lite
```

**rules** — keyword heuristics with confidence 0.4, below the default gate, so everything lands in
`Categories/Unsure`. Use it to rehearse the Gmail plumbing (or in CI) without keys:
`gmail-manager classify --backend rules --confidence 0.2 --dry-run`.

## Tuning

- `--confidence` (0.66) — below this, the message gets `Categories/Unsure` instead of a real label.
- `GMAIL_MANAGER_DEBUG` (empty) — set to `1` in `.env` for per-message progress lines.
- `--body-chars` (1500) — how much message text the model sees. `--no-bodies` sends headers plus the
  Gmail snippet instead of downloading each message body.
- `--workers` (4) — concurrent classifier calls; Gmail calls stay sequential.
- `-q/--query` — any Gmail search string, e.g. `in:inbox newer_than:7d -category:promotions`.
- `-n/--max` (all) — cap on messages per run; paging stops on its own once the inbox is drained, and
  cyclic `nextPageToken` chains are cut after `MAX_REPEATS` repeats of the same message id.

## Development

```bash
uv run pytest        # tests, no network
uv run ruff check src tests
uv run ruff format --check src tests
```

Layout: `gmail_manager/categories.py` (criteria + label names), `state.py` (email → model state),
`classifier.py` (backends + decision parsing), `gmail.py` (OAuth, reads, label writes),
`cli.py` (commands).

## Notes

- `token.json`, `credentials.json`, and `.env` are gitignored.
- Check `--dry-run` output before trusting a new prompt or threshold; `Categories/Unsure` rows show
  where the model was unsure.
- Nothing is cached locally; the mailbox is the source of truth, and already-labelled messages are
  skipped on the next run.
