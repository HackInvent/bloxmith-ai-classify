"""Owned loopback OpenAI-compatible chat server for AI Classify tests. No request leaves loopback; no real key is used.

Replies are deterministic. The server reads the JSON schema the block always states in
its instructions and the text between the block's markers, then answers like a careful
model would: categories whose name appears in the text, keyword sentiment, ``key: value``
lines for extraction and the first words of the text as a summary. Special model names
select failures (invalid JSON, undeclared label, refusal, HTTP errors, slowness...).
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
from types import SimpleNamespace

TEXT = re.compile(r"<(text_[0-9a-f]{10})>\n(.*)\n</\1>", re.S)
POSITIVE = ("great", "thanks", "thank you", "love", "excellent", "perfect", "happy", "merci")
NEGATIVE = ("broken", "angry", "terrible", "late", "refund", "worst", "disappointed", "furious")


def schema_of(system):
    """The reply schema the block writes after 'matching this JSON schema:'."""
    lines = system.split("\n")
    for index, line in enumerate(lines):
        if line.endswith("matching this JSON schema:") and index + 1 < len(lines):
            return json.loads(lines[index + 1])
    raise ValueError("no schema in the instructions")


def text_of(content):
    found = TEXT.search(content)
    if not found:
        raise ValueError("no delimited text")
    return found.group(2)


def words(text, count):
    return " ".join(text.split()[:count])


def _value(raw, schema):
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "number" in kinds:
        try:
            return int(raw) if raw.lstrip("-").isdigit() else float(raw)
        except ValueError:
            return raw
    if "boolean" in kinds:
        return raw.strip().lower() in {"true", "yes", "oui"}
    if "array" in kinds:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def answer(schema, text):
    """A careful deterministic model: only declared names, null when the text says nothing."""
    properties = schema["properties"]
    if "score" in properties:
        names = list(properties["label"]["enum"])
        lowered = text.lower()
        good, bad = sum(word in lowered for word in POSITIVE), sum(word in lowered for word in NEGATIVE)
        if good > bad:
            return {"label": names[0], "score": 0.9, "confidence": 0.8}
        if bad > good:
            return {"label": names[-1], "score": 0.1, "confidence": 0.8}
        return {"label": names[len(names) // 2], "score": 0.5, "confidence": None}
    if "labels" in properties or "label" in properties:
        key = "labels" if "labels" in properties else "label"
        enum = properties["labels"]["items"]["enum"] if key == "labels" else properties["label"]["enum"]
        names = [name for name in enum if name is not None]
        lowered = text.lower()
        found = [name for name in names if name in lowered or name.replace("_", " ") in lowered]
        if key == "labels":
            return {"labels": found, "confidence": 0.75}
        return {"label": found[0] if found else None, "confidence": 0.75 if found else None}
    if "summary" in properties:
        return {"summary": words(text, 8)}
    if "bullets" in properties:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        return {"bullets": [words(sentence, 8) for sentence in sentences[:3]]}
    result = {}
    for key, spec in properties.items():
        match = re.search(rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$", text)
        result[key] = _value(match.group(1), spec) if match else None
    return result


def special(model, schema, text, retry):
    """Content for failure models; None means a normal careful answer."""
    if model == "fixture-invalid" or (model == "fixture-invalid-once" and not retry):
        return "Sure! The category is probably the first one."
    if model == "fixture-unknown" or (model == "fixture-unknown-once" and not retry):
        key = "labels" if "labels" in schema["properties"] else "label"
        return json.dumps({key: ["spam"] if key == "labels" else "spam", "score": 0.5, "confidence": 0.9})
    if model == "fixture-two-labels":
        return json.dumps({"label": ["question", "problem"], "confidence": 0.5})
    if model == "fixture-case":
        value = answer(schema, text)
        if isinstance(value.get("label"), str):
            value["label"] = " " + value["label"].upper() + " "
        return json.dumps(value)
    if model == "fixture-bad-score":
        return json.dumps({**answer(schema, text), "score": 7})
    if model == "fixture-wrong-types":
        return json.dumps({"customer_name": 42, "order_number": "A-17", "amount": "lots", "priority": "urgent",
                           "delivery_date": "next week", "invented": "x"})
    if model == "fixture-fenced":
        return "<think>Reading the categories.</think>\n```json\n" + json.dumps(answer(schema, text)) + "\n```"
    if model == "fixture-huge":
        return json.dumps({"summary": "word " * 20000})
    if model == "fixture-long-summary":
        if "bullets" in schema["properties"]:
            return json.dumps({"bullets": [f"Point {index} " + "detail " * 30 for index in range(10)]})
        return json.dumps({"summary": "word " * 2000})
    return None


@contextmanager
def llm_fixture(*, token=None, tls=False):
    """Serve POST /v1/chat/completions and /v1/responses on a dynamic loopback port."""
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        closing, calls = threading.Event(), []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, code, payload, headers=None):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                if closing.is_set():
                    return
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Connection", "close")
                    for key, value in (headers or {}).items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionError, ssl.SSLError):
                    pass
                finally:
                    self.close_connection = True

            def do_POST(self):
                body = self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 8_388_608))
                auth = self.headers.get("Authorization", "")
                try:
                    request = json.loads(body)
                except ValueError:
                    request = {}
                responses = self.path == "/v1/responses"
                if responses:
                    system, conversation = request.get("instructions", ""), list(request.get("input") or [])
                else:
                    messages = list(request.get("messages") or [])
                    system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
                    conversation = messages[1:]
                retry = any(item.get("role") == "assistant" for item in conversation)
                model = request.get("model")
                record = {"path": self.path, "body": request, "model": model, "retry": retry, "auth_present": bool(auth),
                          "auth_ok": bool(token) and auth == "Bearer " + token, "user_agent": self.headers.get("User-Agent", "")}
                calls.append(record)
                if self.path not in {"/v1/chat/completions", "/v1/responses"}:
                    return self.reply(404, {"error": {"message": "unknown route"}})
                if token and auth != "Bearer " + token:
                    return self.reply(401, {"error": {"message": "Invalid API key provided."}})
                if model == "fixture-slow":
                    closing.wait(30)
                    return self.reply(503, {"error": {"message": "slow fixture released"}})
                if model == "fixture-429":
                    return self.reply(429, {"error": {"message": "Rate limit reached for requests"}}, {"Retry-After": "2"})
                if model == "fixture-503":
                    return self.reply(503, {"error": {"message": "model is loading"}})
                if model == "fixture-missing":
                    return self.reply(404, {"error": {"message": "model 'fixture-missing' not found"}})
                if model == "fixture-echo":
                    return self.reply(400, {"error": {"message": "rejected header " + auth}})
                if model == "fixture-bad-envelope":
                    return self.reply(200, {"object": "chat.completion", "id": "x"})
                if model == "fixture-gzip":
                    return self.reply(200, {"choices": []}, {"Content-Encoding": "gzip"})
                try:
                    schema = schema_of(system)
                    text = text_of(conversation[0]["content"])
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    return self.reply(400, {"error": {"message": f"fixture cannot read the request: {exc}"}})
                record["schema"], record["text"] = schema, text
                refusal = model == "fixture-refusal"
                truncated = model == "fixture-truncated"
                content = None if refusal else special(model, schema, text, retry)
                if content is None and not refusal:
                    content = json.dumps(answer(schema, text), ensure_ascii=False)
                if truncated:
                    content = content[: max(1, len(content) // 2)]
                usage_in, usage_out = len(system + str(conversation)) // 4, len(content or "") // 4
                with_usage = model != "fixture-no-usage"
                if responses:
                    part = ({"type": "refusal", "refusal": "I can't help with that request."} if refusal
                            else {"type": "output_text", "text": content, "annotations": []})
                    payload = {"id": "resp_fixture", "object": "response", "model": model,
                               "status": "incomplete" if truncated else "completed",
                               "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                                          {"type": "message", "id": "msg_1", "role": "assistant", "content": [part]}]}
                    if truncated:
                        payload["incomplete_details"] = {"reason": "max_output_tokens"}
                    if with_usage:
                        payload["usage"] = {"input_tokens": usage_in, "output_tokens": usage_out,
                                            "total_tokens": usage_in + usage_out}
                else:
                    message = {"role": "assistant", "content": content, "refusal": None}
                    if refusal:
                        message = {"role": "assistant", "content": None, "refusal": "I can't help with that request."}
                    payload = {"id": "chatcmpl-fixture", "object": "chat.completion", "model": model,
                               "choices": [{"index": 0, "message": message,
                                            "finish_reason": "length" if truncated else "stop"}]}
                    if with_usage:
                        payload["usage"] = {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                                            "total_tokens": usage_in + usage_out}
                return self.reply(200, payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        ca = ""
        if tls:
            assert shutil.which("openssl"), "OpenSSL is required for the disposable HTTPS fixture"
            ca, key = directory / "ca.pem", directory / "key.pem"
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                            "-keyout", str(key), "-out", str(ca), "-subj", "/CN=localhost",
                            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"], check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(ca, key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            ca = str(ca)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            scheme = "https" if tls else "http"
            yield SimpleNamespace(base_url=f"{scheme}://127.0.0.1:{server.server_port}/v1", ca_file=ca, calls=calls,
                                  port=server.server_port)
        finally:
            closing.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive()
