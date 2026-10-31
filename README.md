# gmail-manager

A CLI that reads your Gmail inbox, classifies each message into **OTP / Primary / Transaction /
Updates / Promotional** with a decision model, and applies the matching **existing** Gmail label.
It never creates labels and never touches Gmail's own category labels.

Two classifier backends, one interface:

| Backend | Engine | Shape of the answer |
| --- | --- | --- |
| `jev` | [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) via `typesafe-sdk` | typed `Choice` + calibrated probabilities + confidence |
| `llm` | any OpenAI-compatible endpoint (your LiteLLM proxy / `pegasus-lite`) via [`system-one-adapter`](https://github.com/typesafe-ai/system-one-adapter-python) | same typed questions, forced JSON answers |
| `rules` | keyword heuristics, no API key or network | fixed 0.4 confidence |

Every classification is one API call that asks two questions in parallel:

1. `category` — a `Choice` over the five categories, each with a written criterion.
2. `delivers_one_time_code` — a `Noul` (probability that the message exists to hand over a 2FA code).

If the noul reaches `--otp-override` (default 0.9) the message is forced to `OTP` regardless of the
choice, because mislabelled security codes are the expensive mistake here.

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
Create them in Gmail first: `OTP`, `Primary`, `Transaction`, `Updates`, `Promotional`.

```bash
uv run gmail-manager labels                      # show the mapping and what's missing
uv run gmail-manager labels --label Updates=News # remap a category to another label name
```

Anything with no matching label is classified and reported, never labelled.

## 4. Classify

```bash
cp .env.example .env                            # then fill in the keys
uv run gmail-manager classify --dry-run -n 20    # nothing is written to the mailbox
uv run gmail-manager classify                    # classify + label 50 newest inbox messages
uv run gmail-manager classify -q "in:inbox is:unread" --confidence 0.7
uv run gmail-manager classify --backend llm --json > run.jsonl
uv run gmail-manager classify --no-bodies -n 200  # headers + snippet only, cheaper and faster
```

Sample output:

```
backend=jev  query='in:inbox'  max=50  mode=apply
classifying 12 messages with jev…

  OTP          p=0.97 c=0.88  GitHub <noreply@github.c  Your GitHub verification code is… applied LL_OTP
  Transaction  p=0.91 c=0.83  Amazon <auto-confirm@amz  Your delivery has arrived        applied LL_TRANSACTION
  Promotional  p=0.88 c=0.79  Brand <offers@brand.com>  48 hours only: 40% off           conf<0.7

12 messages: OTP=3  Transaction=4  Promotional=5  low-confidence=2  skipped(already labelled)=6
```

Messages that already carry one of the managed labels are skipped unless you pass `--reclassify`.

## Backends

`CLASSIFIER_BACKEND` (or `--backend`) picks the engine. With no setting, `jev` is used when
`TYPESAFE_API_KEY` is set and `llm` otherwise.

**Jev** — early-access key from <https://console.typesafe.ai/keys>. Jev answers in parallel, returns
calibrated probabilities and confidence, and cannot produce a value outside the five criteria, which
is exactly what a labelling loop wants.

**LLM** — any OpenAI-compatible chat-completions endpoint. The adapter constrains the model to the
same typed questions (`structured_outputs=True`, per-label probabilities, schema retries). It never
generates free text. Point `LLM_API_BASE` at your LiteLLM proxy:

```dotenv
CLASSIFIER_BACKEND=llm
LLM_API_BASE=http://172.16.1.51:4000/v1
LLM_API_KEY=<your litellm key>
LLM_MODEL=pegasus-lite
```

**rules** — keyword heuristics with confidence 0.4, so the default `--confidence 0.55` gate refuses
to label anything. Use it to rehearse the Gmail plumbing (or in CI) without keys:
`gmail-manager classify --backend rules --confidence 0.2 --dry-run`.

## Tuning

- `--confidence` (0.55) — below this, the answer is reported but not acted on. Raise it to be
  conservative; Jev's confidence is calibrated, so 0.9 means roughly 90% accuracy on that call.
- `--otp-override` (0.9) — the noul threshold that forces `OTP`. Set `1.1` to disable the override.
- `--body-chars` (1500) — how much message text the model sees. `--no-bodies` sends headers plus the
  Gmail snippet instead of downloading each message body.
- `--workers` (4) — concurrent classifier calls; Gmail calls stay sequential.
- `-q/--query` — any Gmail search string, e.g. `in:inbox newer_than:7d -category:promotions`.

## Development

```bash
uv run pytest        # 58 tests, no network
uv run ruff check src tests
uv run ruff format --check src tests
```

Layout: `gmail_manager/categories.py` (criteria + instructions), `state.py` (email → model state),
`classifier.py` (backends + decision combination), `gmail.py` (OAuth, reads, label writes),
`cli.py` (commands).

## Notes

- `token.json`, `credentials.json`, and `.env` are gitignored.
- The `OTP` category catches codes in message bodies; it does not read your real inbox differently
  from the other categories, so a marketing email that *mentions* a code can be pulled into `OTP`.
  Check `--dry-run` output before trusting a new prompt or threshold.
- Nothing is cached locally; the mailbox is the source of truth, and already-labelled messages are
  skipped on the next run.
