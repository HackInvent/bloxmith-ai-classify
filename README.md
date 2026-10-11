# AI Classify

Classify, extract fields, score sentiment or summarize text with an LLM. The block sends
each message to an OpenAI-compatible model you configure (a local Ollama, LM Studio or
vLLM server, or a hosted API) with one bounded request, validates the JSON reply itself,
and then routes the **unchanged** message to the output of the chosen category, or
publishes the extracted fields or the summary. It is not an agent: it calls no tools,
follows no instruction found in the text and never routes anywhere you did not declare.

<!-- block-metadata:start -->
[![Block version: 0.1.0](https://img.shields.io/badge/block-0.1.0-blue)](model.json)
[![BloxSmith compatibility: 1.0.9](https://img.shields.io/badge/BloxSmith-1.0.9-brightgreen)](compatibility.json)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Verified BloxSmith versions: **1.0.9** (bundled-block tests; see [test evidence](compatibility.json)).
<!-- block-metadata:end -->

[![AI Classify — classify, extract, score sentiment or summarize text with an LLM](media/thumbnail.webp)](media/cover.png)

Use it to triage support e-mails into queues, route chat messages by intent, flag
unhappy customers, read an order number and an amount out of free text, or shorten long
documents before a notification. One block, four explicit modes.

## Ports

| Direction | Port | Meaning |
| --- | --- | --- |
| Input | `in` | Text, or JSON whose text is read at `text_path` |
| Output | `result` | JSON document describing the decision, the fields or the summary (every mode) |
| Output | `error` | JSON error document; nothing else is published for that message |
| Output | `unmatched` | The unchanged message when no category fits (`classify`) |
| Output | one per category | The unchanged message, on the output named like the chosen category (`classify`, `sentiment`) |

A new node has the outputs `result`, `error`, `unmatched` and the two example categories
`question` and `problem`. The first three are fixed and addressed by their port number,
so renaming them does not break anything. Category outputs are ordinary user-managed
outputs: **Apply adds an output for every new category**, and you can rename or delete
unused ones in the inspector's Ports tab (up to 40 outputs). Renaming a category in the
editor adds a new output on Apply; the old one stays, still wired, until you delete it.
When a stored category has no output (for example after its output was deleted), the node
card shows "Invalid settings" and the editor offers Apply at once to add it back.

Every message is handled on its own, in One Shot and in Active Runtime. A failed message
is a normal activation that publishes `error`; invalid settings, or a category without an
output, fail the Run before anything is sent.

## Modes

### `classify`

The model chooses among your `categories` (2 to 32, each a `name` and a `description`).
The message is published **unchanged, with its content type**, on the output of the same
name, and `result` describes the decision. When no category fits, the message goes to
`unmatched`. With `multi_label`, every category that applies receives a copy, once each,
in the order of the list; none at all still goes to `unmatched`.

```json
{"mode": "classify", "labels": ["problem"], "unmatched": false, "confidence": 0.82, "multi_label": false,
 "model": "llama3.2", "calls": 1, "retries": 0,
 "usage": {"input_tokens": 312, "output_tokens": 9, "total_tokens": 321}, "text_chars": 118}
```

Category names are output names: 1 to 20 lowercase letters, digits and single
underscores (`billing_question`), unique, and never `result`, `error` or `unmatched`. The
model reads the name **and** the description, so describe each category in plain words.

### `sentiment`

Same routing, with its own list `sentiment_categories` (default `positive`, `neutral`,
`negative`, editable). The model must choose **exactly one** label and a `score` from 0
(most negative) to 1 (most positive), 0.5 being neutral; a reply without a valid score is
invalid. `result` adds `label` and `score`:

```json
{"mode": "sentiment", "label": "negative", "labels": ["negative"], "unmatched": false, "score": 0.08,
 "confidence": 0.9, "model": "llama3.2", "calls": 1, "retries": 0, "text_chars": 64}
```

Switching to `sentiment` and applying adds the `positive`, `neutral` and `negative`
outputs. The classify outputs remain and simply never receive messages in this mode.

### `extract`

Reads the `fields` you declare (1 to 32) into a JSON object. Each field has a `name`
(a JSON key: letter or `_`, then letters, digits or `_`), a `type`, a `description` and a
`required` flag:

| Type | Value |
| --- | --- |
| `string` | Text (whitespace trimmed; an empty text is null) |
| `number` | JSON number; a plain numeric text such as `"42.5"` is accepted, `"1,250"` is not |
| `boolean` | `true` / `false` (also the texts `"true"` / `"false"`) |
| `date` | `YYYY-MM-DD`, checked against the calendar |
| `enum` | One of the field's `values` (matched without case, returned as declared) |
| `list` | Array of texts (empty items dropped, at most 100) |

**Missing values are null, never invented.** A value of the wrong type is also null and
listed in `errors`; a required field that is null is listed in `missing` and `errors`.
Keys the model adds are dropped and only counted. Only `result` is published:

```json
{"mode": "extract", "fields": {"customer_name": "Ada Martin", "order_number": "A-1043", "amount": 89.9,
 "priority": "high", "delivery_date": null}, "valid": true, "missing": [], "errors": [], "ignored_keys": 0,
 "model": "llama3.2", "calls": 1, "retries": 0, "text_chars": 240}
```

Field problems do not trigger a retry: the reply is a usable answer with listed errors.
Use `valid` (for example with a Condition or Switch block) to branch.

### `summarize`

Writes a summary of at most `summary_max_chars` characters, as `bullets` or one
`paragraph`, in the language of the text (`same`), in English (`en`) or in French (`fr`).
A text longer than `chunk_chars` is cut into consecutive parts at paragraph, line,
sentence or word boundaries (at most `max_chunks`); each part is summarized (map), the
partial summaries are combined, in groups when they are still too long (reduce), and a
final request writes the summary in the requested style. Each request is bounded and
counted in `calls`. A summary longer than `summary_max_chars` is cut at a word, or after
the last whole bullet that fits, and `truncated` is true.

```json
{"mode": "summarize", "summary": "- The pump stopped twice on Monday.\n- A spare part was ordered.",
 "bullets": ["The pump stopped twice on Monday.", "A spare part was ordered."], "truncated": false,
 "style": "bullets", "language": "same", "chunks": 3, "model": "llama3.2", "calls": 4, "retries": 0, "text_chars": 31250}
```

## Replies are validated

Model output is untrusted data. The block asks for one JSON object (with a JSON schema
when `structured_output` is on, and always by instruction), then parses and validates the
reply itself: strict JSON (duplicate keys, NaN and trailing text refused; a leading
`<think>…</think>` block or a single Markdown code fence is tolerated), only declared
category names (matched without case, spaces and hyphens counting as `_`), a label count
that fits the mode, a score between 0 and 1, typed fields. An unusable reply is **retried
once** with the reason; if the second reply is unusable too, `error` receives
`invalid_reply` or `unknown_label`. A refusal is reported as `refused` without retry. A
reply cut by `max_output_tokens` counts as invalid. Temperature is 0 by default.

The text is sent between markers derived from its own hash, and the instructions tell the
model that it is data. This reduces, but cannot remove, the influence of instructions
hidden in a text; what such a text can achieve is limited to choosing among your own
categories or filling your own fields.

## Errors

`error` carries `code`, `message`, `retryable`, `mode` and `calls`, plus `http_status`
and a short sanitized `provider_message` for endpoint errors, `detail` for unusable
replies, and the sizes for `input_too_long`. The text and the reply are never copied.

| Code | Meaning |
| --- | --- |
| `input` | Empty text, message not JSON while `text_path` is set, path not found or not a text |
| `input_too_long` | Longer than `max_input_chars`, or needs more than `max_chunks` parts |
| `invalid_reply`, `unknown_label` | Reply unusable, or naming an undeclared category, after one retry |
| `refused` | The model refused the request |
| `secret` | The wallet is locked or `api_key_ref` cannot be resolved |
| `auth`, `rate_limited`, `provider_rejected`, `provider_unavailable`, `http_status` | HTTP 401/403, 408/429, 400/404/413/422, 5xx, other |
| `provider_response` | Malformed, compressed or oversized endpoint response |
| `network`, `timeout` | Connection, DNS, TLS or address policy; `timeout_sec` exceeded |
| `internal` | Unexpected failure; nothing was routed |

HTTP errors and rate limits are not retried automatically; `retryable` tells whether a
later attempt may succeed. Stop aborts a pending request at once and publishes nothing.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `mode` | `classify` | `classify`, `sentiment`, `extract` or `summarize` |
| `categories` | `question`, `problem` | Classify categories, 2–32 `{name, description}` |
| `multi_label` | off | Classify: every category that applies instead of at most one |
| `sentiment_categories` | `positive`, `neutral`, `negative` | Sentiment labels, 2–32 `{name, description}` |
| `fields` | 5 example fields | Extract: 1–32 `{name, type, description, required, values}` |
| `summary_max_chars` | 1200 | Summarize: final summary length, 100–20,000 characters |
| `summary_style` | `bullets` | `bullets` or `paragraph` |
| `summary_language` | `same` | `same`, `en` or `fr` |
| `api_base_url` | `http://127.0.0.1:11434/v1` | OpenAI-compatible API root (see below) |
| `model` | `llama3.2` | Model name sent to the endpoint |
| `api_key_ref` | empty | Optional wallet reference `secret://workspace/<name>`, sent as `Authorization: Bearer` |
| `api_style` | `chat_completions` | `chat_completions` or `responses` |
| `structured_output` | on | Ask for JSON-schema output; the reply is validated either way |
| `temperature` | 0 | 0–2, or empty to not send it |
| `max_output_tokens` | 1024 | 16–65,536 per reply, or 0 to not send a limit |
| `timeout_sec` | 120 | Total time for one message, every request and the retry included, 1–600 |
| `instructions` | empty | Optional context for the model (at most 4,000 characters) |
| `text_path` | empty | Path of the text in a JSON message, e.g. `body.message` or `items[0].text` |
| `max_input_chars` | 100,000 | Longest text accepted, 1–1,000,000 characters |
| `chunk_chars` | 12,000 | Summarize: part size, 1,000–200,000 characters |
| `max_chunks` | 16 | Summarize: most parts per text, 1–64 |
| `allow_remote` | off | Allow a non-loopback endpoint host |
| `allow_insecure_http` | off | Allow plain HTTP to a non-loopback host (never with an API key) |
| `ca_file` | empty | Absolute path of a PEM CA file for an HTTPS endpoint with a private CA |
| `max_response_bytes` | 2 MiB | Largest endpoint response accepted, 64 KiB–16 MiB |

All settings are validated on Apply and before a Run starts, including the lists of the
modes you are not using. Characters are counted as Python characters, not model tokens:
keep `chunk_chars` well below your model's context.

## Model endpoint

`chat_completions` sends `POST <api_base_url>/chat/completions` with a system message and
the text (`response_format: json_schema` when `structured_output` is on, `max_tokens`
when `max_output_tokens` is not 0). `responses` sends `POST <api_base_url>/responses` with
`instructions`, `input`, `store: false`, `text.format: json_schema` and
`max_output_tokens`. Neither streams.

| Server | `api_base_url` | Notes |
| --- | --- | --- |
| Ollama | `http://127.0.0.1:11434/v1` | `ollama pull llama3.2` (or another instruction model) |
| LM Studio | `http://127.0.0.1:1234/v1` | Load a model and start the local server |
| vLLM | `http://127.0.0.1:8000/v1` | Structured output uses vLLM's guided decoding |
| Hosted API | `https://api.openai.com/v1` | Check `allow_remote`; store the key in the wallet and set `api_key_ref` |

If a server rejects `response_format` or `json_schema` (`provider_rejected`), turn
`structured_output` off: the block then relies on its instructions and its own
validation. Some reasoning models refuse `temperature` or `max_tokens`: leave
`temperature` empty, or use `api_style: responses` where the server offers it.

Loopback endpoints (`localhost`, `127.0.0.1`, `::1`) work without extra settings. Any
other host requires `allow_remote`; plain HTTP to it also requires `allow_insecure_http`,
and an API key is then refused. Host names are resolved once per request, every resolved
address is checked, and the connection goes to the checked address: link-local (including
cloud metadata `169.254.169.254`), multicast and unspecified addresses are always
refused. HTTPS verifies the certificate and host name.

## Example

Triage a support inbox by its message body, sending the JSON message unchanged to one of
three queues:

```json
{
  "mode": "classify",
  "categories": [
    {"name": "billing", "description": "Invoices, payments, refunds or prices."},
    {"name": "delivery", "description": "Shipping, parcels, delays or tracking."},
    {"name": "technical", "description": "A product or the website does not work as expected."}
  ],
  "text_path": "body",
  "instructions": "Messages come from the customers of an online bicycle shop."
}
```

Paste these categories in the editor and Apply: the outputs `billing`, `delivery` and
`technical` are added. `{"from": "…", "body": "My parcel is late"}` goes to `delivery`
(only `body` is sent to the model) and `result` holds `{"labels": ["delivery"], …}`. A
message about something else goes to `unmatched`.

## Test with sample text

The editor's **Test with the model** button sends the sample text to the configured
endpoint once, with the settings being edited (unapplied changes included), and shows the
outputs that would receive the message, the score, the extracted fields or the summary,
or the error. Nothing is saved, routed or published, and the endpoint is contacted only
when you click. The same address rules apply. The test is bounded to 20 seconds.

**Limitation:** the editor has no access to the wallet, so an endpoint that needs
`api_key_ref` (most hosted APIs) cannot be tested from the editor; test it with a short
Run instead. Keyless local servers can be tested directly.

## Limits and warnings

- A classification, a score or an extraction is a model's judgment, not a fact. Check
  important decisions, keep `unmatched` wired, and branch on `valid` after `extract`.
- Results vary with the model, its version and the server; temperature 0 makes them more
  repeatable, not identical across servers.
- Hosted endpoints receive the texts you send and may bill for them. Do not send texts
  you are not allowed to share with that provider.
- One message may cause several requests: the retry, and the parts of a long summary.
  `timeout_sec` bounds the total. In managed and linked installations the framework also
  stops a block activation after 300 seconds.
- Classify, sentiment and extract send the whole text in one request: a text larger than
  the model's context fails with `provider_rejected`; summarize it first or use `text_path`.

## Dependencies

Python standard library only (`http.client`, `ssl`, `json`). No SDK is imported, so there
is no `requirements.txt`. The model and its server are installed separately.

## Security notes

- The API key is read from the wallet at run time, only when a request is sent, and only
  over HTTPS or to a loopback host. It is never written to settings, outputs, logs or
  errors; a provider error that echoes it is redacted. `api_key_ref` refuses anything that
  is not a `secret://workspace/<name>` reference.
- Remote endpoints are off by default (`allow_remote`). The block opens no inbound port.
- Traces contain the mode, the routed outputs, counts and error codes, never the text,
  the reply or extracted values. `result` contains what the model returned; treat it as
  untrusted data downstream.
- The node card shows the mode, the category names or summary settings and the model,
  never the endpoint or the reference.
- The editor's sample test sends the sample text to the configured endpoint only when you
  click it, with the same address rules as a Run and no wallet access.

## Tests and qualification

- `tests/F5.01_ai_classify.py`: settings and endpoint policy, category/output contract,
  input and `text_path`, prompts and JSON schemas, strict reply parsing and every field
  type, chunking, the engine with an explicit in-process model double (retry with
  correction, refusal, truncation, unknown labels, wallet handling, cancellation,
  map-reduce request counts and bounds), `execute_runtime` routing and error documents
  with a patched client, Apply adding outputs, validation, the sample test, surfaces and
  EN/FR catalogs, and this README's example.
- `tests/F5.02_ai_classify_runtime.py`: the real HTTP client against
  `tests/llm_fixture.py`, a loopback OpenAI-compatible server with deterministic replies:
  Chat Completions and Responses bodies, Bearer key, structured output, valid replies,
  invalid JSON, unknown label, refusal, truncation, 429/404/503 mapping, echoed-key
  redaction, malformed and compressed responses, timeout and Stop during a pending
  request, HTTPS with a private CA and DNS-answer vetting. Then real graph Runs in One Shot
  and Active Runtime for bundled, managed and linked installations: routing to category
  outputs and `unmatched`, multi-label copies, sentiment, extract, a map-reduce summary,
  error documents, a locked wallet, refusal at preparation of a category without output,
  outputs added through Apply, and Stop.
- `tests/F8.01_ai_classify_ui.py`: managed and linked browser hosts, the mode switch, the
  category and field editors (add, remove, reorder, enum values), live validation, Apply
  adding outputs, Cancel, persistence, the sample test against the loopback fixture,
  English and French, desktop and mobile widths for the modal and the inspector.

**Not qualified against a real model or provider.** Ollama, LM Studio, vLLM and hosted
APIs were not contacted: the protocol is tested with the loopback fixture above, whose
answers are synthetic. Classification and extraction quality, real JSON-schema support of
each server, provider-specific parameters and error bodies, rate limits and billing are
not certified. No real credential was used.

## Framing decisions

These choices complete the block framing where it left options open:

1. **Category name = output name.** A category publishes on the output of the same name;
   fixed outputs (`result` 1, `error` 2, `unmatched` 3) are addressed by port number.
2. **Apply adds missing outputs** for the categories of the active mode, in the same
   operation as the settings; it never renames or deletes an output. A category without
   an output is refused at Run preparation (for example after deleting its output).
3. **Separate lists** for `classify` (`categories`) and `sentiment`
   (`sentiment_categories`), so switching modes keeps both.
4. **Sentiment score** is a polarity from 0 (most negative) to 1 (most positive); it is
   required in the reply. `confidence` is optional in every mode and dropped when invalid.
5. **Retry policy**: exactly one corrective retry for an unusable reply (not JSON, wrong
   shape, undeclared label, wrong label count, truncated); none for refusals, HTTP errors
   or field-level extraction problems.
6. **Lenient but safe matching**: labels match declared names without case; `none`,
   `null`, `unmatched` or `other` mean "no category" unless declared as a category.
7. **Summaries**: bullets are requested as a JSON array and rendered as `- ` lines; the
   final length is enforced by cutting, flagged with `truncated`.
8. **Defaults** target a local Ollama server with `llama3.2`, `chat_completions` and
   structured output on; temperature 0.
9. **The sample test** runs in the editor without wallet access (a framework limit), so it
   only tests endpoints that need no key.

## License

Apache-2.0. See [LICENSE](LICENSE).
