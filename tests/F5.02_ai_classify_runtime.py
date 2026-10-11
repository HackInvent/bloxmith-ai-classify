"""FB1-FB8: real HTTP to a loopback fake model, then real graph Runs in both runtimes and three package origins.

The OpenAI-compatible server is the package-owned fixture on a dynamic loopback port.
Wallets are created in each disposable server; no real provider or credential is used.
"""

import ipaddress
import json
from pathlib import Path
import socket
import sys
import threading
import time
from unittest.mock import patch
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(Path(__file__).resolve().parent)]
from blocs.ai_classify import engine, logic, provider
from blocs.ai_classify.block import AiClassifyBlock
from blocs.ai_classify.logic import ClassifyError
from block_test_packages import install_test_package, surface_payload
from ui_smoke_common import (create_project_api, create_run_api, data_edge, display_node, graph_payload, graph_storage_dir,
                             http_json, isolated_server, prepare_run_api, play_run_api, stop_run_api, text_node,
                             wait_for_run_predicate, wait_for_run_terminal)
from llm_fixture import llm_fixture

TOKEN = "-".join(("fixture", "llm", "token", "not", "real", "4Tz"))
REF = "secret://workspace/llm_fixture"
BLOCK = AiClassifyBlock()
SENTIMENT_PORTS = ((6, "positive"), (7, "neutral"), (8, "negative"))


def refused(call, code):
    try:
        call()
    except ClassifyError as exc:
        assert exc.code == code, (code, exc.code, str(exc))
        return exc
    raise AssertionError("Expected refusal " + code)


def run_once(config, text, token=""):
    """One message through the real HTTP client; returns the outcome and the session."""
    current = engine.Session(config, token=token)
    return engine.process(text, config, current), current


def protocol():
    """# - FB1 - FB3 - FB4 - FB5 - FB6 - request bodies, every mode, invalid replies, HTTP errors, Stop, TLS, DNS policy."""
    defaults = BLOCK.default_config()
    with llm_fixture(token=TOKEN) as api:
        config = logic.configuration({**defaults, "api_base_url": api.base_url, "model": "fixture-llm", "api_key_ref": REF})
        outcome, _ = run_once(config, "There is a problem with my invoice", TOKEN)
        assert outcome["route"] == ["problem"] and outcome["result"]["model"] == "fixture-llm"
        call = api.calls[-1]
        assert call["path"] == "/v1/chat/completions" and call["auth_ok"] and call["user_agent"].startswith("BloxSmith-AI-Classify/")
        body = call["body"]
        assert body["model"] == "fixture-llm" and body["stream"] is False and body["temperature"] == 0 and body["max_tokens"] == 1024
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"] == call["schema"], "the stated and requested schemas agree"
        assert outcome["result"]["usage"]["total_tokens"] > 0
        refused(lambda: run_once(config, "x", ""), "auth")
        refused(lambda: run_once(config, "x", "wrong-token-1"), "auth")
        loose = {**config, "structured_output": False, "temperature": None, "max_output_tokens": 0}
        run_once(loose, "a question", TOKEN)
        assert not {"response_format", "temperature", "max_tokens"} & set(api.calls[-1]["body"])
        responses = {**config, "api_style": "responses"}
        outcome, _ = run_once(responses, "Thanks, it is a question", TOKEN)
        body = api.calls[-1]["body"]
        assert api.calls[-1]["path"] == "/v1/responses" and outcome["route"] == ["question"]
        assert body["store"] is False and body["instructions"].startswith("You classify texts") and body["max_output_tokens"] == 1024
        assert body["input"][0]["role"] == "user" and body["text"]["format"]["name"] == "ai_classify_classify"
        assert outcome["result"]["usage"]["input_tokens"] > 0
        # Each mode over HTTP, with the fixture answering like a careful model.
        multi, _ = run_once({**config, "multi_label": True}, "A question about a problem", TOKEN)
        assert multi["route"] == ["question", "problem"]
        sentiment, _ = run_once({**config, "mode": "sentiment"}, "Thanks, great service!", TOKEN)
        assert sentiment["route"] == ["positive"] and sentiment["result"]["score"] == 0.9
        extract, _ = run_once({**config, "mode": "extract"},
                              "customer_name: Ada Martin\namount: 89.9\npriority: High\ndelivery_date: 2026-02-30", TOKEN)
        fields = extract["result"]
        assert fields["fields"] == {"customer_name": "Ada Martin", "order_number": None, "amount": 89.9, "priority": "high",
                                    "delivery_date": None}
        assert [error["code"] for error in fields["errors"]] == ["invalid_date"] and not fields["valid"]
        long_text = "\n\n".join(f"Chapter {index} explains topic {index}. " + "More words follow here. " * 35 for index in range(5))
        summary, current = run_once({**config, "mode": "summarize", "chunk_chars": 1000, "summary_style": "paragraph"},
                                    long_text, TOKEN)
        assert summary["result"]["chunks"] == 5 and current.calls == 6 and summary["result"]["summary"].startswith("Chapter 0")
        # Unusable replies: one corrective retry, then a coded failure.
        before = len(api.calls)
        assert run_once({**config, "model": "fixture-invalid-once"}, "a question", TOKEN)[0]["result"]["retries"] == 1
        assert [call["retry"] for call in api.calls[before:]] == [False, True]
        for model, code, calls in (("fixture-invalid", "invalid_reply", 2), ("fixture-unknown", "unknown_label", 2),
                                   ("fixture-two-labels", "invalid_reply", 2), ("fixture-truncated", "invalid_reply", 2),
                                   ("fixture-refusal", "refused", 1), ("fixture-bad-score", "invalid_reply", 2)):
            mode = "sentiment" if model == "fixture-bad-score" else "classify"
            current = engine.Session({**config, "model": model, "mode": mode}, token=TOKEN)
            exc = refused(lambda: engine.process("a question", {**config, "model": model, "mode": mode}, current), code)
            assert current.calls == calls and "a question" not in json.dumps(exc.envelope(mode, current.calls)), model
        assert run_once({**config, "model": "fixture-unknown-once"}, "a problem", TOKEN)[0]["route"] == ["problem"]
        assert run_once({**config, "model": "fixture-fenced"}, "a problem", TOKEN)[0]["route"] == ["problem"]
        assert run_once({**config, "model": "fixture-case"}, "a problem", TOKEN)[0]["route"] == ["problem"]
        assert run_once({**config, "model": "fixture-no-usage"}, "a problem", TOKEN)[0]["result"].get("usage") is None
        refused(lambda: run_once({**responses, "model": "fixture-refusal"}, "x", TOKEN), "refused")
        refused(lambda: run_once({**responses, "model": "fixture-truncated"}, "a problem", TOKEN), "invalid_reply")
        # HTTP failures are mapped, never retried, and never leak the key.
        for model, code, retryable in (("fixture-429", "rate_limited", True), ("fixture-503", "provider_unavailable", True),
                                       ("fixture-missing", "provider_rejected", False), ("fixture-bad-envelope", "provider_response", False),
                                       ("fixture-gzip", "provider_response", False)):
            before = len(api.calls)
            exc = refused(lambda: run_once({**config, "model": model}, "x", TOKEN), code)
            assert exc.retryable is retryable and len(api.calls) == before + 1, model
        limited = refused(lambda: run_once({**config, "model": "fixture-429"}, "x", TOKEN), "rate_limited")
        assert limited.details == {"http_status": 429, "provider_message": "Rate limit reached for requests"}
        echoed = refused(lambda: run_once({**config, "model": "fixture-echo"}, "x", TOKEN), "provider_rejected")
        assert TOKEN not in json.dumps(echoed.envelope()) and "[redacted]" in echoed.details["provider_message"]
        refused(lambda: run_once({**config, "model": "fixture-huge", "mode": "summarize", "max_response_bytes": 65536}, "x", TOKEN),
                "provider_response")
        refused(lambda: run_once({**config, "api_base_url": api.base_url + "/other"}, "x", TOKEN), "provider_rejected")
        # - FB8 - Stop and the deadline abort a pending request promptly.
        stop = threading.Event()
        threading.Timer(0.5, stop.set).start()
        began = time.monotonic()
        slow = {**config, "model": "fixture-slow"}
        refused(lambda: engine.process("x", slow, engine.Session(slow, token=TOKEN, cancel=stop.is_set)), "cancelled")
        assert time.monotonic() - began < 3
        began = time.monotonic()
        refused(lambda: engine.process("x", slow, engine.Session(slow, token=TOKEN, deadline=time.monotonic() + 1)), "timeout")
        assert time.monotonic() - began < 3
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    offline = logic.configuration({**defaults, "api_base_url": f"http://127.0.0.1:{closed}/v1"})
    assert refused(lambda: run_once(offline, "x"), "network").retryable
    # - FB7 - HTTPS verifies the certificate chain and host name; a private CA is opt-in through ca_file.
    with llm_fixture(tls=True) as api:
        secure = logic.configuration({**defaults, "api_base_url": api.base_url, "model": "fixture-llm", "ca_file": api.ca_file})
        assert run_once(secure, "a problem")[0]["route"] == ["problem"]
        named = {**secure, "api_base_url": f"https://localhost:{api.port}/v1"}
        assert run_once(named, "a question")[0]["route"] == ["question"]
        refused(lambda: run_once({**secure, "ca_file": ""}, "x"), "network")
    # - FB7 - DNS answers are vetted before any socket: no rebinding of localhost, metadata or link-local targets.
    private = str(ipaddress.ip_network(((192 << 24) | (168 << 16), 16))[7])
    answers = lambda address: [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 8000))]
    local = logic.configuration({**defaults, "api_base_url": "http://localhost:8000/v1", "api_key_ref": REF})
    remote = logic.configuration({**local, "allow_remote": True})
    insecure = logic.configuration({**remote, "allow_insecure_http": True})
    metadata = logic.configuration({**defaults, "api_base_url": "https://llm.example.test/v1", "allow_remote": True})
    for settings, address, message in ((local, private, "allow_remote"), (remote, private, "allow_insecure_http"),
                                       (insecure, private, "never sent"), (metadata, "169.254.169.254", "allow_remote")):
        with patch.object(provider.socket, "getaddrinfo", return_value=answers(address)), \
                patch.object(provider.socket, "socket", side_effect=AssertionError("socket opened")):
            assert message in str(refused(lambda: run_once(settings, "x", TOKEN), "network"))
    print("[ok] AI Classify model protocol", flush=True)


# --------------------------------------------------------------------------- graph runs


def aic_node(node_id, y, version, config=None, extra=()):
    node = BLOCK.build_node_payload(node_id=node_id, position={"x": 380, "y": y})
    node["config"] = {**BLOCK.default_config(), **(config or {})}
    for port_id, name in extra:
        node["outputs"].append({"id": port_id, "name": name, "title": name, "emits": ["message/*"], "multiplicity": "many"})
    if version:
        node["block_version"] = version
    return node


def chain(prefix, message, version, y, config=None, extra=(), wire=()):
    """source -> AI Classify -> a display on each output named in ``wire`` (published values are read from run state).

    Few displays keep the number of concurrently scheduled nodes low in One Shot runs.
    """
    node = aic_node(f"{prefix}-aic", y, version, config, extra)
    nodes = [text_node(f"{prefix}-src", prefix, message, 80, y), node]
    edges = [data_edge(f"{prefix}-in", f"{prefix}-src", 1, node["id"], 1)]
    for index, port in enumerate(node["outputs"]):
        if port["name"] not in wire:
            continue
        target = f"{prefix}-{port['name']}"
        nodes.append(display_node(target, port["name"], 760, y - 60 + 30 * index))
        edges.append(data_edge(f"{prefix}-e-{port['name']}", node["id"], port["id"], target, 1))
    return nodes, edges


def document(title, *chains):
    nodes, edges = [], []
    for chain_nodes, chain_edges in chains:
        nodes += chain_nodes
        edges += chain_edges
    return graph_payload(title, nodes, edges)


def published(run, node_id, ports):
    """Values actually published by one node, keyed by output name."""
    outputs = run.get("output_values") or {}
    return {name: outputs[f"{node_id}:{port}"] for port, name in ports
            if (outputs.get(f"{node_id}:{port}") or {}).get("value")}


def received(run, node_id):
    text = str(((run.get("worker_rows") or {}).get(node_id) or {}).get("received") or "")
    return "" if text == "-" else " ".join(text.split())


BASE_PORTS = ((1, "result"), (2, "error"), (3, "unmatched"), (4, "question"), (5, "problem"))
MESSAGE = json.dumps({"from": "customer@example.test", "body": "My invoice has a problem"})
LONG = "\n\n".join(f"Chapter {index} explains topic {index}. " + "More words follow here. " * 35 for index in range(4))


def success_graph(base_url, version):
    """# - FB1 - FB2 - FB3 - FB4 - FB5 - FB6 - every mode in one Run; routed payloads unchanged, documents on result/error."""
    common = {"api_base_url": base_url, "model": "fixture-llm"}
    return document(
        "AI Classify modes",
        chain("inv", MESSAGE, version, 80, {**common, "text_path": "body"}, wire=("problem", "question")),
        chain("hello", "Hello there, nice weather today.", version, 300, common),
        chain("multi", "A question about a problem", version, 520, {**common, "multi_label": True}),
        chain("mood", "Thanks, great service!", version, 740, {**common, "mode": "sentiment"}, SENTIMENT_PORTS),
        chain("form", "customer_name: Ada Martin\norder_number: A-1043\namount: 89.9\npriority: High", version, 1060,
              {**common, "mode": "extract"}),
        chain("long", LONG, version, 1280, {**common, "mode": "summarize", "chunk_chars": 1000, "summary_style": "paragraph"}),
        chain("bad", "a question", version, 1500, {**common, "model": "fixture-invalid"}, wire=("question",)),
        chain("refuse", "a problem", version, 1720, {**common, "model": "fixture-refusal", "api_style": "responses"}))


def check_success(server, api, mode, version, origin):
    created = create_run_api(server, success_graph(api.base_url, version), runtime_mode=mode)
    run = wait_for_run_terminal(server, created["run_id"], timeout_sec=120)
    where = (mode, origin)
    assert run["status"] == "success", (where, run.get("logs"))
    invoice = published(run, "inv-aic", BASE_PORTS)
    assert sorted(invoice) == ["problem", "result"], (where, invoice)
    assert invoice["problem"]["value"] == MESSAGE and invoice["problem"]["content_type"] == "text/plain"
    assert "My invoice has a problem" in received(run, "inv-problem") and not received(run, "inv-question")
    assert json.loads(invoice["result"]["value"])["labels"] == ["problem"]
    assert invoice["result"]["content_type"] == "application/json"
    hello = published(run, "hello-aic", BASE_PORTS)
    assert sorted(hello) == ["result", "unmatched"] and hello["unmatched"]["value"] == "Hello there, nice weather today."
    assert json.loads(hello["result"]["value"])["unmatched"] is True
    multi = published(run, "multi-aic", BASE_PORTS)
    assert sorted(multi) == ["problem", "question", "result"], (where, multi)
    mood = published(run, "mood-aic", (*BASE_PORTS, *SENTIMENT_PORTS))
    assert sorted(mood) == ["positive", "result"] and json.loads(mood["result"]["value"])["score"] == 0.9, (where, mood)
    form = published(run, "form-aic", BASE_PORTS)
    assert sorted(form) == ["result"]
    fields = json.loads(form["result"]["value"])
    assert fields["fields"] == {"customer_name": "Ada Martin", "order_number": "A-1043", "amount": 89.9, "priority": "high",
                                "delivery_date": None} and fields["valid"], fields
    summary = json.loads(published(run, "long-aic", BASE_PORTS)["result"]["value"])
    assert summary["chunks"] == 4 and summary["calls"] == 5 and summary["summary"].startswith("Chapter 0"), summary
    bad = published(run, "bad-aic", BASE_PORTS)
    assert sorted(bad) == ["error"] and json.loads(bad["error"]["value"])["code"] == "invalid_reply", (where, bad)
    assert json.loads(bad["error"]["value"])["calls"] == 2 and not received(run, "bad-question")
    refusal = published(run, "refuse-aic", BASE_PORTS)
    assert sorted(refusal) == ["error"] and json.loads(refusal["error"]["value"])["code"] == "refused"
    logs = "\n".join(run.get("logs") or [])
    for line in ("[ai-classify] classify: routed to problem (1 request(s)).",
                 "[ai-classify] classify: routed to unmatched (1 request(s)).",
                 "[ai-classify] classify: routed to question, problem (1 request(s)).",
                 "[ai-classify] sentiment: positive, score 0.9, routed to positive (1 request(s)).",
                 "[ai-classify] extract: 5 field(s), valid (1 request(s)).",
                 "[ai-classify-error] classify failed with invalid_reply after 2 request(s); published on error.",
                 "[ai-classify-error] classify failed with refused after 1 request(s); published on error."):
        assert line in logs, (where, line, logs[-4000:])
    own = [line for line in run.get("logs") or [] if line.startswith("[ai-classify")]
    assert not any("invoice" in line or "Ada" in line for line in own), "traces never contain the text"


def check_prepare(server, api, mode, version, origin):
    """# - FB7 - a category without an output is refused at preparation, before any request."""
    before = len(api.calls)
    graph = document("AI Classify invalid", chain("nope", "a question", version, 80, {
        "api_base_url": api.base_url, "model": "fixture-llm",
        "categories": [{"name": "question", "description": ""}, {"name": "vip", "description": ""}]}))
    created = create_run_api(server, graph, runtime_mode=mode)
    run = wait_for_run_terminal(server, created["run_id"], timeout_sec=60)
    assert run["status"] == "failed", (mode, origin, run.get("logs"))
    logs = "\n".join(run.get("logs") or [])
    assert "[prepare-error] nope-aic" in logs and "Invalid AI Classify settings: categories 2: Category vip has no output" in logs, logs[-3000:]
    assert len(api.calls) == before and not published(run, "nope-aic", BASE_PORTS)


def start(server, project, graph, mode):
    """Project-scoped Run (the wallet is resolved for saved projects)."""
    (graph_storage_dir(server, project) / "graph.json").write_text(json.dumps({**graph, "graph_id": project["graph_id"]}),
                                                                 encoding="utf-8")
    http_json(server.base_url, f"/api/projects/{project['graph_id']}/graph/reload", method="POST", payload={})
    return create_run_api(server, graph, project_id=project["graph_id"], runtime_mode=mode)["run_id"]


def check_wallet(server, keyed, mode, version, origin):
    """# - FB7 - FB8 - the Bearer key comes from the wallet, never reaches run state; a locked wallet sends nothing."""
    graph = document("AI Classify wallet", chain("key", "Where is my question answered?", version, 80, {
        "api_base_url": keyed.base_url, "model": "fixture-llm", "api_key_ref": REF}))
    project = create_project_api(server, document=graph)["project"]
    http_json(server.base_url, "/api/application/secrets/init", method="POST", payload={"password": "disposable-llm-fixture"})
    http_json(server.base_url, "/api/application/secrets", method="POST", payload={"name": "llm_fixture", "value": TOKEN})
    before = len(keyed.calls)
    run = wait_for_run_terminal(server, start(server, project, graph, mode), timeout_sec=60)
    assert run["status"] == "success", (mode, origin, run.get("logs"))
    assert sorted(published(run, "key-aic", BASE_PORTS)) == ["question", "result"]
    assert keyed.calls[before]["auth_ok"] and TOKEN not in json.dumps(run), "the wallet key leaked into the Run state"
    http_json(server.base_url, "/api/application/secrets/lock", method="POST", payload={})
    calls = len(keyed.calls)
    locked = wait_for_run_terminal(server, start(server, project, graph, mode), timeout_sec=60)
    error = published(locked, "key-aic", BASE_PORTS)
    assert sorted(error) == ["error"] and json.loads(error["error"]["value"])["code"] == "secret", (mode, origin, error)
    assert len(keyed.calls) == calls, "no request without the key"


def check_stop(server, api, mode, version, origin):
    """# - FB8 - Stop while the model request is pending: the Run ends promptly and nothing is published."""
    graph = document("AI Classify stop", chain("slow", "a question", version, 80, {
        "api_base_url": api.base_url, "model": "fixture-slow", "timeout_sec": 120}))
    if mode == "zeromq_active":
        run_id = prepare_run_api(server, graph)["run_id"]
        play_run_api(server, run_id)
    else:
        run_id = create_run_api(server, graph, runtime_mode=mode)["run_id"]
    wait_for_run_predicate(server, run_id, lambda state: state.get("node_statuses", {}).get("slow-aic") == "running",
                           "The AI Classify node never started running.", timeout_sec=60)
    time.sleep(0.8)
    began = time.monotonic()
    stop_run_api(server, run_id)
    final = wait_for_run_predicate(server, run_id, lambda state: state.get("status") in {"cancelled", "failed", "success"},
                                   "The stopped Run did not end.", timeout_sec=30)
    assert time.monotonic() - began < 15, "Stop did not abort the pending request promptly"
    assert final["status"] == "cancelled", (mode, origin, final.get("status"), final.get("logs"))
    assert not published(final, "slow-aic", BASE_PORTS), final.get("output_values")


def check_apply_and_preview(server, api, model, version):
    """# - FB7 - FB8 - Apply through the real API adds the outputs of new categories; the sample test calls the endpoint."""
    key = f"{model['kind']}@{model['version']}" if version else "ai_classify"
    node = aic_node("apply-aic", 80, version, {"api_base_url": api.base_url, "model": "fixture-llm"})
    graph = document("AI Classify apply", ([text_node("apply-src", "Mood", "Thanks, great service!", 80, 80), node],
                                           [data_edge("apply-in", "apply-src", 1, "apply-aic", 1)]))
    project = create_project_api(server, document=graph)["project"]
    state = http_json(server.base_url, f"/api/projects/{project['project_id']}/graph/state")
    current = next(item for item in state["document"]["nodes"] if item["id"] == "apply-aic")
    applied = http_json(server.base_url, f"/api/blocks/{quote(key, safe='')}/ui-action", method="POST", payload={
        "project_id": project["project_id"], "base_version": state["version"], "apply_graph_patch": True,
        "op_id_prefix": "aic-apply", "node": current, "action": "save_settings",
        "values": {"node_patch": {"config": {"mode": "sentiment"}}}})
    assert applied.get("graph_patch", {}).get("ok") is True and applied["created_outputs"] == ["positive", "neutral", "negative"], applied
    updated = http_json(server.base_url, f"/api/projects/{project['project_id']}/graph/state")["document"]
    stored = next(item for item in updated["nodes"] if item["id"] == "apply-aic")
    assert [(port["id"], port["name"]) for port in stored["outputs"]] == [*BASE_PORTS, *SENTIMENT_PORTS]
    assert stored["config"]["mode"] == "sentiment"
    again = http_json(server.base_url, f"/api/blocks/{quote(key, safe='')}/ui-action", method="POST", payload={
        "project_id": project["project_id"], "apply_graph_patch": True, "node": stored, "action": "save_settings",
        "values": {"node_patch": {"config": {"mode": "sentiment"}}}})
    assert "created_outputs" not in again, "existing outputs are never added twice"
    for port_id, name in SENTIMENT_PORTS:
        updated["nodes"].append(display_node(f"apply-{name}", name, 760, 40 * port_id))
        updated["edges"].append(data_edge(f"apply-e-{name}", "apply-aic", port_id, f"apply-{name}", 1))
    run = wait_for_run_terminal(server, create_run_api(server, updated, runtime_mode="centralized")["run_id"], timeout_sec=60)
    assert run["status"] == "success" and sorted(published(run, "apply-aic", (*BASE_PORTS, *SENTIMENT_PORTS))) == ["positive", "result"]
    preview = lambda values: http_json(server.base_url, f"/api/blocks/{quote(key, safe='')}/preview", method="POST",
                                       payload={"node": stored, "values": values})
    before = len(api.calls)
    shown = preview({"config": {"mode": "classify"}, "sample": "I have a problem"})
    assert shown["outcome"] == "ok" and shown["outputs"] == ["problem"] and len(api.calls) == before + 1, shown
    assert preview({"config": {"api_key_ref": REF}, "sample": "x"})["sample_error"]["code"] == "preview_wallet"
    assert preview({"config": {"model": "fixture-429"}, "sample": "x"})["error"]["code"] == "rate_limited"
    checked = http_json(server.base_url, f"/api/blocks/{quote(key, safe='')}/validate", method="POST",
                        payload={"node": stored, "values": {"config": {"mode": "classify"}}})
    assert checked["valid"] and checked["unused_outputs"] == ["positive", "neutral", "negative"]


def check_surfaces(server, model, version, base_url):
    """# - FB8 - installed surfaces render without template residue, the endpoint or the wallet reference on the card."""
    node = aic_node("surface-aic", 80, version, {"api_base_url": base_url, "api_key_ref": REF})
    for surface in ("modal", "inspector_panel", "node_card"):
        html = surface_payload(server, model, node, surface=surface)["html"]
        assert "{{" not in html and "__title__" not in html and TOKEN not in html
        if surface == "node_card":
            assert "question · problem" in html and REF not in html and base_url not in html
        else:
            assert html.count("data-aic-item>") == 13 and "data-aic-test" in html and "data-owned-apply" in html


def runs():
    with llm_fixture() as api, llm_fixture(token=TOKEN) as keyed:
        for origin in (None, "managed", "linked"):
            with isolated_server() as server:
                version, model = None, None
                if origin:
                    model = install_test_package(server, "ai_classify", origin=origin)
                    version = model["version"]
                    check_surfaces(server, model, version, api.base_url)
                for mode in ("centralized", "zeromq_active"):
                    check_success(server, api, mode, version, origin)
                    check_prepare(server, api, mode, version, origin)
                check_wallet(server, keyed, "zeromq_active" if origin else "centralized", version, origin)
                if origin != "linked":
                    check_apply_and_preview(server, api, model or BLOCK.model, version)
                check_stop(server, api, "zeromq_active", version, origin)
                if origin is None:
                    check_stop(server, api, "centralized", version, origin)
            print(f"[ok] AI Classify runtime ({origin or 'bundled'})", flush=True)


def main():
    protocol()
    runs()


if __name__ == "__main__":
    main()
