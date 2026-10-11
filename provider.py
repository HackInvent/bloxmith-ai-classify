"""OpenAI-compatible chat client: one configured endpoint, vetted addresses, bounded and cancellable.

Only the Python standard library is used. Each HTTP exchange runs in a short-lived
daemon thread so Stop or the activation deadline can abort the socket at once. The
reply envelope (Chat Completions or Responses) is decoded here; the content of the
model's answer is validated by ``logic``.
"""

import http.client
import ipaddress
import json
from pathlib import Path
import socket
import ssl
import threading
import time

from .logic import ClassifyError, endpoint

USER_AGENT = "BloxSmith-AI-Classify/0.1.0"
MAX_ERROR_BODY = 65_536
MAX_RETRY_ASSISTANT = 4000


def remaining(deadline):
    """Every socket operation shares the activation deadline."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise ClassifyError("timeout", "The model request exceeded timeout_sec.", retryable=True)
    return max(0.001, left)


def _address(value):
    address = ipaddress.ip_address(str(value).split("%")[0])
    return getattr(address, "ipv4_mapped", None) or address


def address_allowed(value, config):
    """Loopback always; other addresses only with allow_remote, never link-local/metadata or multicast."""
    try:
        address = _address(value)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if config.get("allow_remote") is not True:
        return False
    return not (address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved)


def provider_message(raw, token):
    """A short single-line provider explanation, with the credential removed if echoed."""
    text = raw.decode("utf-8", errors="replace")
    message = text
    try:
        document = json.loads(text)
        if isinstance(document, dict):
            error = document.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
            elif isinstance(error, str):
                message = error
            elif isinstance(document.get("message"), str):
                message = document["message"]
            elif isinstance(document.get("detail"), str):
                message = document["detail"]
    except ValueError:
        pass
    message = " ".join(str(message).split())
    if token:
        message = message.replace(token, "[redacted]")
    return message[:300]


def _status_error(status, raw, token):
    detail = {"http_status": status}
    explanation = provider_message(raw, token)
    if explanation:
        detail["provider_message"] = explanation
    if status in {401, 403}:
        return ClassifyError("auth", "The endpoint refused the credentials; check api_key_ref.", **detail)
    if status in {408, 429}:
        return ClassifyError("rate_limited", "The endpoint is rate limiting or timed out; retry later.",
                             retryable=True, **detail)
    if status >= 500:
        return ClassifyError("provider_unavailable", "The endpoint failed or is unavailable.", retryable=True, **detail)
    if status in {400, 404, 413, 422}:
        return ClassifyError("provider_rejected", "The endpoint rejected the request; check api_base_url, model, "
                             "api_style and structured_output.", **detail)
    return ClassifyError("http_status", "The endpoint returned an unexpected HTTP status.", **detail)


def _post(target, config, body, token, deadline, holder):
    """One exchange on a vetted numeric address; DNS is resolved once and never re-resolved."""
    host, port = target["host"], target["port"]
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except OSError:
        raise ClassifyError("network", "The endpoint host name could not be resolved.", retryable=True) from None
    if not infos or len(infos) > 32 or any(not address_allowed(item[4][0], config) for item in infos):
        raise ClassifyError("network", "The endpoint host resolves to an address this block may not contact; "
                            "non-loopback servers need allow_remote.")
    local = all(_address(item[4][0]).is_loopback for item in infos)
    if target["scheme"] == "http" and not local:
        if config.get("allow_insecure_http") is not True:
            raise ClassifyError("network", "Plain HTTP to a non-loopback address requires allow_insecure_http.")
        if token:
            raise ClassifyError("network", "The API key is never sent over plain HTTP to a non-loopback address.")
    context = None
    if target["scheme"] == "https":
        ca = config.get("ca_file") or ""
        if ca:
            path = Path(ca)
            if not path.is_file() or path.stat().st_size > 1_048_576:
                raise ClassifyError("network", "The configured ca_file is missing, not a file or too large.")
            context = ssl.create_default_context(cafile=str(path))
        else:
            context = ssl.create_default_context()
    sock = None
    for family, kind, proto, _, address in infos[:4]:
        candidate = socket.socket(family, kind, proto)
        holder["sock"] = candidate
        try:
            candidate.settimeout(remaining(deadline))
            candidate.connect(address)
            sock = candidate
            break
        except ClassifyError:
            candidate.close()
            raise
        except OSError:
            candidate.close()
    if sock is None:
        raise ClassifyError("network", "Unable to connect to the endpoint.", retryable=True)
    try:
        if context is not None:
            sock.settimeout(remaining(deadline))
            sock = context.wrap_socket(sock, server_hostname=host)
            holder["sock"] = sock
        connection = http.client.HTTPConnection(host, port, timeout=remaining(deadline))
        connection.sock = sock
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Accept-Encoding": "identity", "User-Agent": USER_AGENT, "Connection": "close"}
        if token:
            headers["Authorization"] = "Bearer " + token
        sock.settimeout(remaining(deadline))
        connection.request("POST", target["path"], body=body, headers=headers)
        sock.settimeout(remaining(deadline))
        response = connection.getresponse()
        encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
        if encoding != "identity":
            raise ClassifyError("provider_response", "Compressed responses are not supported; identity encoding was requested.")
        if response.status != 200:
            raise _status_error(response.status, response.read(MAX_ERROR_BODY), token)
        limit = config["max_response_bytes"]
        length = response.getheader("Content-Length")
        if length is not None and (not length.strip().isdigit() or int(length) > limit):
            raise ClassifyError("provider_response", f"The response exceeds max_response_bytes ({limit}).")
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ClassifyError("provider_response", f"The response exceeds max_response_bytes ({limit}).")
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise ClassifyError("provider_response", "The endpoint response is not valid JSON.") from None
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _abort(holder):
    """Unblock the worker thread; the half-finished exchange is discarded."""
    sock = holder.get("sock")
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def call(target, config, body, token, cancel, deadline):
    """Run one exchange in a daemon thread and stop waiting on Stop or the deadline."""
    box, holder = {}, {}

    def work():
        try:
            box["value"] = _post(target, config, body, token, deadline, holder)
        except BaseException as exc:  # noqa: BLE001 - transported to the caller below.
            box["error"] = exc

    thread = threading.Thread(target=work, name="ai-classify-request", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.05)
        if not thread.is_alive():
            break
        if cancel():
            _abort(holder)
            raise ClassifyError("cancelled", "Stopped while waiting for the model; nothing was published.")
        if time.monotonic() >= deadline:
            _abort(holder)
            raise ClassifyError("timeout", "The model request exceeded timeout_sec.", retryable=True)
    error = box.get("error")
    if isinstance(error, ClassifyError):
        raise error
    if isinstance(error, socket.timeout):
        raise ClassifyError("timeout", "The model request exceeded timeout_sec.", retryable=True)
    if isinstance(error, ssl.SSLError):
        raise ClassifyError("network", "TLS negotiation or certificate verification failed.")
    if isinstance(error, http.client.HTTPException):
        raise ClassifyError("provider_response", "The endpoint returned a malformed HTTP response.", retryable=True)
    if isinstance(error, OSError):
        raise ClassifyError("network", "The connection to the endpoint failed.", retryable=True)
    if error is not None:
        raise ClassifyError("internal", "The model request failed unexpectedly.")
    return box["value"]


# --------------------------------------------------------------------------- request bodies


def chat_body(config, request):
    """POST /chat/completions body: system message, conversation, optional JSON schema."""
    body = {"model": config["model"], "stream": False,
            "messages": [{"role": "system", "content": request["system"]}, *request["messages"]]}
    if config["temperature"] is not None:
        body["temperature"] = config["temperature"]
    if config["max_output_tokens"]:
        body["max_tokens"] = config["max_output_tokens"]
    if config["structured_output"]:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": request["schema_name"], "schema": request["schema"], "strict": True}}
    return body


def responses_body(config, request):
    """POST /responses body: instructions, conversation input, optional JSON schema; never stored."""
    body = {"model": config["model"], "instructions": request["system"], "input": list(request["messages"]),
            "store": False}
    if config["temperature"] is not None:
        body["temperature"] = config["temperature"]
    if config["max_output_tokens"]:
        body["max_output_tokens"] = config["max_output_tokens"]
    if config["structured_output"]:
        body["text"] = {"format": {"type": "json_schema", "name": request["schema_name"],
                                   "schema": request["schema"], "strict": True}}
    return body


# --------------------------------------------------------------------------- reply envelopes


def _usage(raw, prompt_key, completion_key):
    if not isinstance(raw, dict):
        return None
    numbers = {}
    for public, key in (("input_tokens", prompt_key), ("output_tokens", completion_key), ("total_tokens", "total_tokens")):
        value = raw.get(key)
        if type(value) is int and 0 <= value < 10**12:
            numbers[public] = value
    if "total_tokens" not in numbers and {"input_tokens", "output_tokens"} <= set(numbers):
        numbers["total_tokens"] = numbers["input_tokens"] + numbers["output_tokens"]
    return numbers or None


def _model(payload, config):
    model = payload.get("model") if isinstance(payload, dict) else None
    return model if isinstance(model, str) and 0 < len(model) <= 200 and model.isprintable() else config["model"]


def chat_reply(payload, config):
    """Decode a Chat Completions response into text, refusal, truncation, model and usage."""
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ClassifyError("provider_response", "The chat completion has no choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ClassifyError("provider_response", "The chat completion has no message.")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict) and isinstance(part.get("text"), str))
    if content is not None and not isinstance(content, str):
        raise ClassifyError("provider_response", "The chat completion message content is not text.")
    refusal = message.get("refusal")
    return {"text": content, "refusal": refusal if isinstance(refusal, str) and refusal.strip() else None,
            "truncated": choices[0].get("finish_reason") == "length", "model": _model(payload, config),
            "usage": _usage(payload.get("usage"), "prompt_tokens", "completion_tokens")}


def responses_reply(payload, config):
    """Decode a Responses API response: output_text parts, refusal parts and incomplete status."""
    if not isinstance(payload, dict):
        raise ClassifyError("provider_response", "The response is not a JSON object.")
    if payload.get("status") == "failed" or payload.get("error"):
        raise ClassifyError("provider_unavailable", "The endpoint reported a failed response.", retryable=True)
    output = payload.get("output")
    if not isinstance(output, list):
        raise ClassifyError("provider_response", "The response has no output list.")
    texts, refusal = [], None
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str) and part["refusal"].strip():
                refusal = part["refusal"]
    truncated = payload.get("status") == "incomplete"
    return {"text": "".join(texts) if texts else None, "refusal": refusal, "truncated": truncated,
            "model": _model(payload, config), "usage": _usage(payload.get("usage"), "input_tokens", "output_tokens")}


def complete(config, request, *, token="", cancel=lambda: False, deadline=None):
    """Send one request and return the decoded reply envelope; HTTP and network failures raise ClassifyError."""
    target = endpoint(config)
    deadline = deadline if deadline is not None else time.monotonic() + config["timeout_sec"]
    if cancel():
        raise ClassifyError("cancelled", "Stopped before the model request; nothing was published.")
    responses = config["api_style"] == "responses"
    body = responses_body(config, request) if responses else chat_body(config, request)
    raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    payload = call(target, config, raw, token, cancel, deadline)
    return responses_reply(payload, config) if responses else chat_reply(payload, config)


def retry_request(request, reply_text, detail, correction):
    """The same request plus the unusable answer (bounded) and a correction message."""
    previous = (reply_text or "")[:MAX_RETRY_ASSISTANT] or "(empty reply)"
    return {**request, "messages": [*request["messages"], {"role": "assistant", "content": previous},
                                    {"role": "user", "content": correction(detail)}]}
