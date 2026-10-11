"""AI Classify: classify, extract fields, score sentiment or summarize text with an OpenAI-compatible model."""

import json
import time

from bloxsmith_app.block_api import (APPLICATION_JSON, BlockDefinition, BlockRuntimeOutput, BlockRuntimePreparation,
                                     BlockRuntimeResult, TEXT_PLAIN)
from . import engine, logic, ui
from .logic import ClassifyError, SettingsError

ROUTED_EMITS = ["message/*", "application/json", "text/plain"]


# FB1 - classify: one bounded request; the unchanged message goes to the output of the chosen category, result describes it.
# FB2 - multi_label sends a copy to every chosen category output; a message that fits no category goes to unmatched.
# FB3 - sentiment: exactly one declared label, routed like a category, plus a validated score from 0 to 1.
# FB4 - extract: declared fields only, typed and validated; missing values are null, errors listed, nothing invented.
# FB5 - summarize: bounded chunks summarized then combined (map-reduce), style, language and maximum length enforced.
# FB6 - Model replies are untrusted: strict JSON parsing, one corrective retry, then invalid_reply/unknown_label/refused errors.
# FB7 - Validate settings, endpoint policy, wallet reference and category outputs on Apply (adding missing outputs) and at preparation.
# FB8 - Run identically in centralized and zeromq_active for every package origin; Stop aborts a pending request; translated editor.
class AiClassifyBlock(BlockDefinition):
    """Routing decisions come only from declared categories; model output is data, never a command."""

    kind = "ai_classify"

    def ui_assets(self, surface="modal"):
        """Every installation origin uses the static release manifest."""
        return list(self.model.get("ui_assets", {}).get(surface, []))

    # ------------------------------------------------------------------ runtime

    def prepare_runtime(self, context):
        """Refuse invalid settings and categories without an output before any activation; no network or wallet."""
        try:
            config = logic.configuration(context.config)
        except SettingsError as exc:
            raise ValueError(f"Invalid AI Classify settings: {exc.text()}") from None
        pairs = logic.port_pairs(getattr(context, "output_ports", None))
        if pairs is not None:
            report = logic.output_report(config, pairs)
            errors = report["errors"] + logic.missing_output_errors(config, report["missing"])
            if errors:
                raise ValueError(f"Invalid AI Classify settings: {errors[0].text()}")
        return BlockRuntimePreparation()

    def execute_runtime(self, context):
        """Analyze one message; publish routed copies, result, or one error document."""
        try:
            config = logic.configuration(context.config)
            report = logic.output_report(config, logic.port_pairs(context.output_ports))
            errors = report["errors"] + logic.missing_output_errors(config, report["missing"])
            if errors:
                raise errors[0]
        except SettingsError as exc:
            message = self._settings_message(exc)
            return BlockRuntimeResult(status="failed", outputs=[], error=message, exit_code=1,
                                      logs=[f"[ai-classify-error] Invalid AI Classify settings: {exc.text()} Nothing was published."])
        if not context.has_input_value("in", "1"):
            return BlockRuntimeResult(status="skipped", outputs=[], logs=["[ai-classify] No message to analyze."])
        raw = context.input_value("in", "1")
        content_type = self._content_type(context)
        cancel = context.services.get("cancel_requested", lambda: False)
        session = engine.Session(config, resolve_secret=lambda reference: context.resolve_secret(reference), cancel=cancel)
        mode = config["mode"]
        try:
            outcome = engine.process(raw, config, session)
        except ClassifyError as exc:
            if exc.code == "cancelled" or cancel():
                return self._stopped(context, session)
            envelope = exc.envelope(mode, session.calls)
            log = (f"[ai-classify-error] {mode} failed with {exc.code} after {session.calls} request(s); "
                   "published on error.")
            return self._publish(context, {logic.ERROR_PORT: envelope}, (), "", content_type, log,
                                 {"mode": mode, "status": "error", "code": exc.code, "calls": session.calls})
        except (ValueError, TypeError, OSError, RuntimeError, RecursionError, MemoryError):
            # Raw driver, provider or credential details never reach outputs.
            envelope = ClassifyError("internal", "The message could not be analyzed; nothing was routed.").envelope(
                mode, session.calls)
            return self._publish(context, {logic.ERROR_PORT: envelope}, (), "", content_type,
                                 f"[ai-classify-error] {mode} failed with internal; published on error.",
                                 {"mode": mode, "status": "error", "code": "internal", "calls": session.calls})
        if cancel():
            return self._stopped(context, session)
        result = outcome["result"]
        names = [*outcome["route"], *(["unmatched"] if outcome["unmatched"] else [])]
        log = self._decision_log(result, names)
        return self._publish(context, {logic.RESULT_PORT: result}, names, self._message_text(raw), content_type, log,
                             {"mode": mode, "status": "ok", "outputs": names, "calls": result["calls"]})

    def _stopped(self, context, session):
        return BlockRuntimeResult(status="skipped", outputs=self._clear(context),
                                  logs=[f"[ai-classify] Stopped after {session.calls} request(s); nothing was published."])

    @staticmethod
    def _message_text(raw):
        """The routed payload: the received text unchanged, or compact JSON for a native value."""
        if isinstance(raw, str):
            return raw
        try:
            return logic.dumps(raw)
        except (TypeError, ValueError, RecursionError):
            return str(raw)

    @staticmethod
    def _content_type(context):
        """Source content type: the delivering event keeps it in both runtimes; the attribute is the fallback."""
        port_ids = {int(port.id) for port in context.input_ports if str(port.name) == "in"} or {logic.IN_PORT}
        events = [event for event in (context.input_events or ()) if int(event.input_port_id) in port_ids]
        if len(events) == 1 and str(events[0].content_type or "").strip():
            return str(events[0].content_type)
        return context.input_content_type("in", "1", default=TEXT_PLAIN)

    @staticmethod
    def _clear(context):
        return [BlockRuntimeOutput(port_id=int(port.id), port_name=str(port.name), value="", emitted=False)
                for port in context.output_ports]

    def _publish(self, context, documents, routed, payload, content_type, log, summary):
        """Emit documents on fixed outputs (by id) and the message on routed outputs (by name); clear the others."""
        chosen = set(routed)
        outputs = []
        for port in context.output_ports:
            ident, name = int(port.id), str(port.name)
            if ident in documents:
                text = context.redact_secrets(json.dumps(documents[ident], ensure_ascii=False, allow_nan=False))
                outputs.append(BlockRuntimeOutput(port_id=ident, port_name=name, value=text, content_type=APPLICATION_JSON))
            elif (ident == logic.UNMATCHED_PORT and "unmatched" in chosen) or (
                    ident not in logic.FIXED_PORTS and name in chosen):
                outputs.append(BlockRuntimeOutput(port_id=ident, port_name=name, value=payload, content_type=content_type))
            else:
                outputs.append(BlockRuntimeOutput(port_id=ident, port_name=name, value="", emitted=False))
        return BlockRuntimeResult(status="success", outputs=outputs, content_type=APPLICATION_JSON,
                                  last_message=logic.dumps(summary), worker_received=", ".join(routed) or summary["status"],
                                  metadata={self.kind: summary}, logs=[context.redact_secrets(log)])

    @staticmethod
    def _decision_log(result, names):
        """English trace of the decision, without the text, the labels' evidence or the summary."""
        calls = f"{result['calls']} request(s)" + (f", {result['retries']} retry" if result["retries"] else "")
        mode = result["mode"]
        if mode == "classify":
            return f"[ai-classify] classify: routed to {', '.join(names)} ({calls})."
        if mode == "sentiment":
            return f"[ai-classify] sentiment: {result['label']}, score {result['score']}, routed to {names[0]} ({calls})."
        if mode == "extract":
            state = "valid" if result["valid"] else f"{len(result['errors'])} field error(s)"
            return f"[ai-classify] extract: {len(result['fields'])} field(s), {state} ({calls})."
        cut = ", truncated" if result.get("truncated") else ""
        return (f"[ai-classify] summarize: {len(result['summary'])} characters from {result['chunks']} chunk(s){cut} "
                f"({calls}).")

    # ------------------------------------------------------------------ editor

    def _settings_message(self, error):
        """Translated, position-aware message for settings refused on Apply or at run time."""
        detail = self.translate(f"block.ai_classify.err_{error.code}", error.params, fallback=error.message)
        if error.list and error.index:
            detail = self.translate("block.ai_classify.invalid_item", {"list": error.list, "index": error.index,
                                                                      "reason": detail}, fallback="{list} {index}: {reason}")
        return self.translate("block.ai_classify.invalid_settings", {"detail": detail},
                              fallback="Invalid AI Classify settings: {detail}")

    def handle_ui_action(self, *, node, action, values, payload=None):
        """Validate and canonicalize the draft; Apply also adds the outputs of new categories."""
        created = []
        if action in {"save_settings", "modal_update_fields", "inspector_update_fields"}:
            patch = dict((values or {}).get("node_patch") or {})
            if "config" in patch:
                stored = node.get("config") if isinstance(node.get("config"), dict) else {}
                draft = patch["config"] if isinstance(patch["config"], dict) else {}
                try:
                    config = logic.configuration({**stored, **logic.editor_settings(draft)})
                    pairs = logic.node_pairs(node)
                    if pairs is not None:
                        report = logic.output_report(config, pairs)
                        if report["errors"]:
                            raise report["errors"][0]
                        created = report["missing"]
                        if len(pairs) + len(created) > logic.MAX_OUTPUTS:
                            raise SettingsError("outputs_too_many", "This block can have at most {max} outputs; remove "
                                                "unused outputs in the Ports tab first.", {"max": logic.MAX_OUTPUTS},
                                                field="outputs")
                except SettingsError as exc:
                    return {"error": self._settings_message(exc)}
                patch["config"] = logic.to_config(config)
                values = {**(values or {}), "node_patch": patch}
        result = super().handle_ui_action(node=node, action="modal_update_fields" if action == "save_settings" else action,
                                          values=values, payload=payload)
        result = ui.replace_config(result, node)
        if created:
            node_id = str(node.get("id") or "")
            ports = [{"op": "create_port", "node_id": node_id, "direction": "output", "name": name, "title": name,
                      "emits": list(ROUTED_EMITS), "multiplicity": "many"} for name in created]
            result["graph_operations"] = ports + list(result.get("graph_operations") or [])
            result["created_outputs"] = list(created)
        return result

    def handle_ui_request(self, *, node, route, method, values, payload=None):
        """Read-only draft checks; the sample test calls the endpoint only on an explicit click."""
        draft = (values or {}).get("config")
        draft = draft if isinstance(draft, dict) else {}
        stored = node.get("config") if isinstance(node.get("config"), dict) else {}
        if route == "validate":
            return ui.validation(draft, stored, logic.node_pairs(node))
        if route == "preview":
            return self._preview(draft, (values or {}).get("sample"), stored, node)
        return super().handle_ui_request(node=node, route=route, method=method, values=values, payload=payload)

    def _preview(self, draft, sample, stored, node):
        checked = ui.validation(draft, stored, logic.node_pairs(node))
        if not checked["valid"]:
            return checked
        config = logic.configuration({**stored, **logic.editor_settings(draft)})
        if not isinstance(sample, str) or not sample.strip():
            return {**checked, "sample_error": {"code": "sample_empty", "params": {}, "message": "Type a sample text first."}}
        if len(sample) > logic.MAX_SAMPLE_CHARS:
            return {**checked, "sample_error": {"code": "sample_too_large", "params": {"max": logic.MAX_SAMPLE_CHARS},
                                                "message": f"The sample is longer than {logic.MAX_SAMPLE_CHARS} characters."}}
        if config["api_key_ref"]:
            return {**checked, "sample_error": {"code": "preview_wallet", "params": {}, "message":
                    "The editor cannot read the wallet, so an endpoint that needs api_key_ref can only be tested in a Run."}}
        deadline = time.monotonic() + min(config["timeout_sec"], logic.PREVIEW_DEADLINE_SEC)
        session = engine.Session(config, deadline=deadline, token="")
        try:
            outcome = engine.process(sample, config, session)
        except ClassifyError as exc:
            return {**checked, "outcome": "error", "error": exc.envelope(config["mode"], session.calls)}
        except (ValueError, TypeError, OSError, RuntimeError, RecursionError, MemoryError):
            return {**checked, "outcome": "error", "error": ClassifyError(
                "internal", "The sample could not be analyzed.").envelope(config["mode"], session.calls)}
        names = [*outcome["route"], *(["unmatched"] if outcome["unmatched"] else [])]
        return {**checked, "outcome": "ok", "result": outcome["result"], "outputs": names}

    def render_modal(self, *, node, payload=None):
        """Render the owned editor inside the opaque modal contract."""
        return ui.modal(self, node, payload)

    def render_inspector_panel(self, *, node, payload=None):
        """Keep the same editor usable in the narrow inspector, with the Ports tab."""
        return ui.inspector(self, node, payload)

    def render_node_card(self, *, node, payload=None):
        """Show the mode, its categories or summary settings and the model; never the endpoint or the reference."""
        return ui.card(self, node)
