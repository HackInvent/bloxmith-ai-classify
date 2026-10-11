"""One message, one bounded analysis: text, model request(s) with one retry, validation and routing decision."""

import time

from . import logic, provider
from .logic import ClassifyError, ReplyError

FAILED_REPLY = {
    "invalid_reply": "The model reply could not be used, even after one retry.",
    "unknown_label": "The model named a category that is not declared, even after one retry.",
}


class Session:
    """Wallet token, cancellation, deadline and counters shared by the requests of one message.

    ``llm`` replaces the HTTP client in tests: ``llm(request, config, deadline=, cancel=)``
    returns the same reply envelope as :func:`provider.complete`.
    """

    def __init__(self, config, *, resolve_secret=None, cancel=None, deadline=None, llm=None, token=None):
        self.config = config
        self.cancel = cancel or (lambda: False)
        self.deadline = deadline if deadline is not None else time.monotonic() + config["timeout_sec"]
        self._resolve, self._token, self._llm = resolve_secret, token, llm
        self.calls = self.retries = 0
        self.usage = None
        self.model = config["model"]

    def token(self):
        """Resolve the wallet reference only when a request is actually sent."""
        ref = self.config["api_key_ref"]
        if not ref:
            return ""
        if self._token is None:
            if not callable(self._resolve):
                raise ClassifyError("secret", "The wallet service is unavailable; api_key_ref cannot be resolved.")
            try:
                value = self._resolve(ref)
            except Exception:  # noqa: BLE001 - locked/missing wallet details are not exposed.
                raise ClassifyError("secret", "The wallet secret is unavailable: unlock the wallet and check api_key_ref.") from None
            if not isinstance(value, str) or not 1 <= len(value) <= 8192 or any(ord(c) < 33 or ord(c) > 126 for c in value):
                raise ClassifyError("secret", "The wallet secret must be a printable API key without spaces.")
            self._token = value
        return self._token

    def send(self, request):
        if self.cancel():
            raise ClassifyError("cancelled", "Stopped before the model request; nothing was published.")
        provider.remaining(self.deadline)
        token = self.token()
        self.calls += 1
        if self._llm is not None:
            reply = self._llm(request, self.config, deadline=self.deadline, cancel=self.cancel)
        else:
            reply = provider.complete(self.config, request, token=token, cancel=self.cancel, deadline=self.deadline)
        self._account(reply)
        return reply

    def _account(self, reply):
        if isinstance(reply.get("model"), str) and reply["model"]:
            self.model = reply["model"]
        usage = reply.get("usage")
        if isinstance(usage, dict) and usage:
            total = dict(self.usage or {})
            for key, value in usage.items():
                total[key] = total.get(key, 0) + value
            self.usage = total

    @staticmethod
    def _accept(task, reply):
        if reply.get("refusal"):
            raise ClassifyError("refused", "The model refused to process this text.")
        if reply.get("truncated"):
            raise ReplyError("invalid_reply", "The reply was cut off by max_output_tokens.")
        return task.parse(logic.decode_reply(reply.get("text")))

    def ask(self, task):
        """One request; an unusable reply is retried once with a correction, then fails with its code."""
        request = task.request()
        reply = self.send(request)
        try:
            return self._accept(task, reply)
        except ReplyError as first:
            self.retries += 1
            reply = self.send(provider.retry_request(request, reply.get("text"), str(first), logic.correction))
            try:
                return self._accept(task, reply)
            except ReplyError as second:
                raise ClassifyError(second.code, FAILED_REPLY[second.code], detail=str(second)) from None

    def stats(self):
        result = {"model": self.model, "calls": self.calls, "retries": self.retries}
        if self.usage:
            result["usage"] = dict(self.usage)
        return result


def _summarize(text, config, session):
    limit, style = config["summary_max_chars"], config["summary_style"]
    pieces = logic.chunks(text, config["chunk_chars"])
    if len(pieces) > config["max_chunks"]:
        raise ClassifyError("input_too_long", f"The text needs {len(pieces)} chunks; max_chunks is {config['max_chunks']}.",
                            chunks=len(pieces), max_chunks=config["max_chunks"])
    if len(pieces) == 1:
        parsed = session.ask(logic.summary_task(config, pieces[0], limit=limit, style=style))
        return logic.fit_summary(parsed, style, limit), 1
    # Map: each chunk gets a bounded partial summary; reduce: combine groups until they fit one request.
    target = max(200, min(limit, config["chunk_chars"] // len(pieces)))
    partials = []
    for index, piece in enumerate(pieces, start=1):
        parsed = session.ask(logic.summary_task(config, piece, limit=target, part=(index, len(pieces))))
        partials.append(logic.clip(parsed["summary"], target * 2)[0])
    for _level in range(logic.MAX_REDUCE_LEVELS):
        if len(partials) < 2 or len("\n\n".join(partials)) <= config["chunk_chars"]:
            break
        packed = logic.groups(partials, config["chunk_chars"])
        if len(packed) == len(partials):
            break
        combined = []
        for group in packed:
            if len(group) == 1:
                combined.append(group[0])
                continue
            parsed = session.ask(logic.summary_task(config, "\n\n".join(group), limit=target, combine=True))
            combined.append(logic.clip(parsed["summary"], target * 2)[0])
        partials = combined
    parsed = session.ask(logic.summary_task(config, "\n\n".join(partials), limit=limit, style=style, combine=True))
    return logic.fit_summary(parsed, style, limit), len(pieces)


def process(raw, config, session):
    """Analyze one message. Returns ``{"result": document, "route": [category names], "unmatched": bool}``.

    ``route`` is empty for extract and summarize. Raises ClassifyError; nothing is published then.
    """
    text = logic.message_text(raw, config)
    mode = config["mode"]
    route, unmatched = [], False
    if mode == "classify":
        parsed = session.ask(logic.classify_task(config, text))
        route = parsed["labels"]
        unmatched = not route
        document = {"mode": mode, "labels": route, "unmatched": unmatched, "confidence": parsed["confidence"],
                    "multi_label": config["multi_label"]}
    elif mode == "sentiment":
        parsed = session.ask(logic.sentiment_task(config, text))
        route = parsed["labels"]
        document = {"mode": mode, "label": route[0], "labels": route, "unmatched": False, "score": parsed["score"],
                    "confidence": parsed["confidence"]}
    elif mode == "extract":
        parsed = session.ask(logic.extract_task(config, text))
        document = {"mode": mode, **parsed}
    else:
        fitted, count = _summarize(text, config, session)
        document = {"mode": mode, **fitted, "style": config["summary_style"], "language": config["summary_language"],
                    "chunks": count}
    document.update(session.stats())
    document["text_chars"] = len(text)
    return {"result": document, "route": route, "unmatched": unmatched}
