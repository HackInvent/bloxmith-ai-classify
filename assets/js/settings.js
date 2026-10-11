/** AI Classify editor: drafts stay local until Apply; checks and the sample test call the block's own routes. */
const KEY = "block.ai_classify.";
const NAME = /^[a-z0-9]+(?:_[a-z0-9]+)*$/;
const RESERVED = new Set(["result", "error", "unmatched"]);
const ROUTING = { classify: "categories", sentiment: "sentiment_categories" };
const OWNER = { categories: "classify", sentiment_categories: "sentiment", fields: "extract", multi_label: "classify",
  summary_max_chars: "summarize", summary_style: "summarize", summary_language: "summarize" };

export function bindEditor(root, api) {
  const editor = root.querySelector("[data-aic-editor]");
  const apply = root.querySelector("[data-owned-apply]");
  if (!editor || !apply) return () => {};
  const controller = new AbortController();
  const { signal } = controller;
  const status = root.querySelector("[data-aic-status]");
  const groupError = editor.querySelector("[data-aic-group-error]");
  const outputsStatus = editor.querySelector("[data-aic-outputs-status]");
  const sample = editor.querySelector("[data-aic-sample]");
  const test = editor.querySelector("[data-aic-test]");
  const result = editor.querySelector("[data-aic-preview-result]");
  const parse = (value, fallback) => { try { return JSON.parse(value || ""); } catch { return fallback; } };
  const outputs = new Set(parse(editor.dataset.aicOutputs, []));
  const fixed = new Set(parse(editor.dataset.aicFixed, []));
  // IDs are DOM references for aria-describedby, not security tokens.
  const base = `aic-${Math.random().toString(36).slice(2)}`;
  let serial = 0, dirty = false, pending = false, generation = 0, timer = 0, checkSeq = 0, testSeq = 0;
  const readOnly = () => Boolean(api.isReadOnly?.());

  /** Translated text that follows a language change; fallback is the English text. */
  const say = (element, key, fallback, params = {}) => {
    if (!element) return;
    if (globalThis.CWI18n?.set) globalThis.CWI18n.set(element, KEY + key, params, fallback);
    else element.textContent = api.t?.(KEY + key, params, fallback) || fallback;
  };
  /** Raw content (names, values and model output) never carries a translation marker. */
  const raw = (element, text) => {
    if (!element) return;
    if (globalThis.CWI18n?.setText) globalThis.CWI18n.setText(element, text);
    else element.textContent = text;
  };
  const make = (tag, className, parent) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    parent?.append(element);
    return element;
  };
  const lists = () => [...editor.querySelectorAll("[data-aic-list]")];
  const list = name => editor.querySelector(`[data-aic-list="${name}"]`);
  const rowsOf = container => [...container.querySelector("[data-aic-rows]").querySelectorAll(":scope > [data-aic-item]")];
  const field = (row, name) => row.querySelector(`[data-aic-field="${name}"]`);
  const mode = () => editor.querySelector('[data-aic-setting="mode"]:checked')?.value || "classify";
  const validName = value => NAME.test(value) && value.length <= 20 && !RESERVED.has(value) && !fixed.has(value);

  function syncFieldRow(row) {
    const values = row.querySelector("[data-aic-values-field]");
    if (values) values.hidden = field(row, "type")?.value !== "enum";
  }

  function prepareRow(row) {
    const error = row.querySelector("[data-aic-item-error]");
    if (!error.id) error.id = `${base}-${++serial}`;
    if (field(row, "type")) syncFieldRow(row);
  }

  /** Positions, counts and the availability of every list action. */
  function renumber() {
    for (const container of lists()) {
      const rows = rowsOf(container);
      const min = Number(container.dataset.min) || 0, max = Number(container.dataset.max) || 32;
      rows.forEach((row, index) => {
        raw(row.querySelector("[data-aic-number]"), String(index + 1));
        row.querySelector('[data-aic-move="up"]').disabled = readOnly() || index === 0;
        row.querySelector('[data-aic-move="down"]').disabled = readOnly() || index === rows.length - 1;
        row.querySelector("[data-aic-remove]").disabled = readOnly() || rows.length <= min;
      });
      container.querySelector("[data-aic-add]").disabled = readOnly() || rows.length >= max;
      raw(container.querySelector("[data-aic-count]"), `${rows.length} / ${max}`);
    }
  }

  /** Per-category output state and the outputs Apply adds, from the node's current outputs. */
  function syncOutputs() {
    for (const name of Object.values(ROUTING)) {
      for (const row of rowsOf(list(name))) {
        const state = row.querySelector("[data-aic-output-state]");
        const value = field(row, "name").value.trim();
        if (!validName(value)) {
          delete state.dataset.state;
          raw(state, "");
        } else if (outputs.has(value)) {
          state.dataset.state = "exists";
          say(state, "output_exists", "Output ready");
        } else {
          state.dataset.state = "new";
          say(state, "output_new", "Output added on Apply");
        }
      }
    }
    const active = ROUTING[mode()];
    if (!active) {
      say(outputsStatus, "outputs_fixed_mode", "In this mode only result and error receive messages.");
      return;
    }
    const names = rowsOf(list(active)).map(row => field(row, "name").value.trim()).filter(validName);
    const missing = [...new Set(names.filter(name => !outputs.has(name)))];
    const unused = [...outputs].filter(name => !names.includes(name));
    const params = { missing: missing.join(", "), unused: unused.join(", ") };
    if (missing.length && unused.length) {
      say(outputsStatus, "outputs_add_unused", "Apply adds the outputs {missing}. Outputs no category uses: {unused}; delete or rename them in the Ports tab.", params);
    } else if (missing.length) {
      say(outputsStatus, "outputs_add", "Apply adds the outputs {missing}.", params);
    } else if (unused.length) {
      say(outputsStatus, "outputs_unused", "Outputs no category uses: {unused}. They never receive messages; delete or rename them in the Ports tab.", params);
    } else {
      raw(outputsStatus, "");
    }
  }

  function syncMode() {
    const current = mode();
    for (const section of editor.querySelectorAll("[data-aic-mode-section]")) section.hidden = section.dataset.aicModeSection !== current;
    const help = editor.querySelector("[data-aic-routing-help]");
    if (help) help.hidden = !ROUTING[current];
    syncOutputs();
  }

  function sync() {
    apply.disabled = !dirty || pending || readOnly();
  }

  /** The draft sent to the server; it converts enum values and validates everything, exactly like Apply. */
  function draft() {
    const config = {};
    for (const control of editor.querySelectorAll("[data-aic-setting]")) {
      const key = control.dataset.aicSetting;
      if (control.type === "radio") {
        if (control.checked) config[key] = control.value;
      } else if (control.type === "checkbox") {
        config[key] = control.checked;
      } else if (["integer", "number"].includes(control.dataset.aicType)) {
        const text = control.value.trim();
        const number = Number(text);
        if (!text && control.dataset.aicNullable === "true") config[key] = null;
        else config[key] = text !== "" && Number.isFinite(number) ? number : control.value;
      } else {
        config[key] = control.value;
      }
    }
    for (const name of Object.values(ROUTING)) {
      config[name] = rowsOf(list(name)).map(row => ({ name: field(row, "name").value.trim(), description: field(row, "description").value }));
    }
    config.fields = rowsOf(list("fields")).map(row => ({
      name: field(row, "name").value.trim(),
      type: field(row, "type").value,
      required: field(row, "required").checked,
      description: field(row, "description").value,
      values_text: field(row, "values_text").value,
    }));
    return config;
  }

  const slots = () => [groupError, ...editor.querySelectorAll("[data-aic-item-error], [data-aic-list-error], [data-aic-setting-error]")];

  function clearErrors() {
    const ids = new Set();
    for (const slot of slots()) {
      slot.hidden = true;
      raw(slot, "");
      if (slot.id) ids.add(slot.id);
    }
    for (const control of editor.querySelectorAll("[aria-invalid]")) {
      control.removeAttribute("aria-invalid");
      const kept = (control.getAttribute("aria-describedby") || "").split(" ").filter(id => id && !ids.has(id));
      if (kept.length) control.setAttribute("aria-describedby", kept.join(" "));
      else control.removeAttribute("aria-describedby");
    }
  }

  /** The first problem per slot: a row, a list, a setting, or the general line for anything else. */
  function showErrors(errors) {
    clearErrors();
    const hiddenModes = new Set();
    for (const error of errors || []) {
      let target = null, control = null;
      const owner = OWNER[error.list || error.field];
      if (owner && owner !== mode()) {
        hiddenModes.add(owner);
        continue;
      }
      if (error.list && error.index) {
        const row = rowsOf(list(error.list))[error.index - 1];
        if (!row) continue;
        target = row.querySelector("[data-aic-item-error]");
        if (error.key) control = field(row, error.key === "values" ? "values_text" : error.key);
      } else if (error.list || list(error.field || "")) {
        target = list(error.list || error.field).querySelector("[data-aic-list-error]");
      } else if (error.field) {
        target = editor.querySelector(`[data-aic-setting-error="${error.field}"]`);
        control = editor.querySelector(`[data-aic-setting="${error.field}"]:not([type="radio"])`);
      }
      target = target || groupError;
      if (!target.hidden) continue;
      if (!target.id) target.id = `${base}-${++serial}`;
      target.hidden = false;
      say(target, `err_${error.code}`, error.message, error.params || {});
      if (target.closest("details:not([open])")) target.closest("details").open = true;
      if (control) {
        control.setAttribute("aria-invalid", "true");
        control.setAttribute("aria-describedby", `${control.getAttribute("aria-describedby") || ""} ${target.id}`.trim());
      }
    }
    if (hiddenModes.size && groupError.hidden) {
      groupError.hidden = false;
      say(groupError, "hidden_mode_error", "The settings of mode {modes} need a fix: select that mode to see them.",
        { modes: [...hiddenModes].join(", ") });
    }
  }

  /** Server-side validation of the current draft, ignoring answers to older drafts. */
  async function check() {
    const sequence = ++checkSeq;
    const response = await api.blockRequest("validate", { payload: { values: { config: draft() } } });
    if (signal.aborted || sequence !== checkSeq) return null;
    showErrors(response?.errors || []);
    return response;
  }

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(() => { check().catch(() => {}); }, 350);
  }

  function changed() {
    generation += 1;
    checkSeq += 1;
    dirty = true;
    raw(status, "");
    sync();
    syncOutputs();
    schedule();
  }

  function addRow(container) {
    const rows = rowsOf(container);
    if (readOnly() || rows.length >= (Number(container.dataset.max) || 32)) return;
    const row = container.querySelector("template[data-aic-template]").content.firstElementChild.cloneNode(true);
    container.querySelector("[data-aic-rows]").append(row);
    prepareRow(row);
    globalThis.CWI18n?.apply?.(row);
    renumber();
    changed();
    field(row, "name").focus();
  }

  /** Move a row among its siblings and keep the focus on a usable control. */
  function move(row, direction) {
    const sibling = direction === "up" ? row.previousElementSibling : row.nextElementSibling;
    if (!sibling) return;
    if (direction === "up") sibling.before(row); else sibling.after(row);
    renumber();
    changed();
    const button = row.querySelector(`[data-aic-move="${direction}"]`);
    (button.disabled ? row.querySelector(`[data-aic-move="${direction === "up" ? "down" : "up"}"]`) : button).focus();
  }

  function remove(row, container) {
    if (rowsOf(container).length <= (Number(container.dataset.min) || 0)) return;
    const next = row.nextElementSibling || row.previousElementSibling;
    row.remove();
    renumber();
    changed();
    (next?.querySelector("[data-aic-remove]:not(:disabled)") || container.querySelector("[data-aic-add]")).focus();
  }

  async function save() {
    if (!dirty || pending || readOnly()) return;
    const started = generation;
    pending = true;
    sync();
    say(status, "checking", "Checking the settings…");
    try {
      let verdict = null;
      try { verdict = await check(); } catch { verdict = null; }
      if (signal.aborted) return;
      if (verdict && !verdict.valid) {
        say(status, "fix_errors", "Not applied: fix the highlighted settings first.");
        return;
      }
      const title = root.querySelector("[data-block-title-field]");
      const response = await api.applyAction("save_settings", { node_patch: { ...(title ? { title: title.value } : {}), config: draft() } });
      if (response?.error) throw new Error(response.error);
      if (signal.aborted) return;
      const created = Array.isArray(response?.created_outputs) ? response.created_outputs : [];
      for (const name of created) outputs.add(name);
      if (started === generation) dirty = false;
      if (created.length) say(status, "saved_outputs", "Settings applied. Outputs added: {outputs}.", { outputs: created.join(", ") });
      else say(status, "saved", "Settings applied.");
      syncOutputs();
    } catch (error) {
      if (!signal.aborted) raw(status, String(error?.message || error));
    } finally {
      pending = false;
      if (!signal.aborted) sync();
    }
  }

  function line(parent, key, fallback, params) {
    const element = make("p", "", parent);
    say(element, key, fallback, params);
    return element;
  }

  function renderRouting(document, routed, head) {
    if (document.unmatched) {
      say(head, "preview_unmatched", "No category fits: the message would be sent to unmatched.");
    } else {
      say(head, "preview_routed", "The message would be sent to {outputs}.", { outputs: routed.join(", ") });
    }
    if (document.mode === "sentiment") line(result, "preview_score", "Sentiment score: {score}", { score: String(document.score) });
    if (document.confidence !== null && document.confidence !== undefined) {
      line(result, "preview_confidence", "Confidence: {confidence}", { confidence: String(document.confidence) });
    }
  }

  function renderExtract(document, head) {
    const problems = (document.errors || []).length;
    if (problems) say(head, "preview_extract_errors", "Extracted fields, with {count} problems", { count: problems });
    else say(head, "preview_extract", "Extracted fields");
    const table = make("dl", "", result);
    for (const [name, value] of Object.entries(document.fields || {})) {
      raw(make("dt", "", table), name);
      raw(make("dd", "", table), JSON.stringify(value));
    }
    if (problems) {
      const items = make("ul", "", result);
      for (const problem of document.errors) {
        const item = make("li", "", items);
        raw(make("code", "", item), problem.field);
        item.append(" ");
        say(make("span", "", item), `field_${problem.code}`, problem.message);
      }
    }
  }

  function renderSummary(document, head) {
    say(head, "preview_summary", "Summary ({count} characters)", { count: (document.summary || "").length });
    raw(make("pre", "aic-summary", result), document.summary || "");
    if (document.truncated) line(result, "preview_truncated", "The reply was longer than summary_max_chars and was cut.");
  }

  async function preview() {
    const sequence = ++testSeq;
    result.replaceChildren();
    delete result.dataset.outcome;
    delete result.dataset.outputs;
    const head = make("p", "aic-preview-head", result);
    say(head, "testing", "Asking the model…");
    test.disabled = true;
    try {
      const response = await api.blockRequest("preview", { payload: { values: { config: draft(), sample: sample.value } } });
      if (signal.aborted || sequence !== testSeq) return;
      if (!response?.valid) {
        result.dataset.outcome = "invalid";
        showErrors(response?.errors || []);
        say(head, "preview_invalid", "Fix the highlighted settings to test them.");
        return;
      }
      showErrors(response.errors || []);
      if (response.sample_error) {
        result.dataset.outcome = "invalid";
        say(head, `err_${response.sample_error.code}`, response.sample_error.message, response.sample_error.params || {});
        return;
      }
      if (response.outcome === "error") {
        const error = response.error || {};
        result.dataset.outcome = "error";
        result.dataset.code = error.code || "";
        say(head, `code_${error.code}`, error.message || "The request failed.");
        if (error.http_status || error.provider_message) {
          raw(make("p", "field-hint", result), [error.http_status ? `HTTP ${error.http_status}` : "", error.provider_message || ""]
            .filter(Boolean).join(": "));
        }
        return;
      }
      const document = response.result || {};
      const routed = response.outputs || [];
      result.dataset.outcome = "ok";
      result.dataset.outputs = routed.join(" ");
      if (["classify", "sentiment"].includes(document.mode)) renderRouting(document, routed, head);
      else if (document.mode === "extract") renderExtract(document, head);
      else renderSummary(document, head);
      line(result, "preview_meta", "Model {model} · {count} requests", { model: document.model || "", count: document.calls || 0 })
        .className = "field-hint";
    } catch {
      if (signal.aborted || sequence !== testSeq) return;
      result.dataset.outcome = "unavailable";
      say(head, "preview_unavailable", "The test could not run.");
    } finally {
      if (!signal.aborted && sequence === testSeq) test.disabled = readOnly();
    }
  }

  for (const container of lists()) for (const row of rowsOf(container)) prepareRow(row);
  renumber();
  syncMode();
  const initial = parse(editor.dataset.aicInitial, {});
  showErrors(initial.errors || []);
  // Stored categories without their outputs: Apply is offered at once to add them.
  if ((initial.missing_outputs || []).length && !readOnly()) dirty = true;
  const watched = "[data-aic-field], [data-aic-setting], [data-block-title-field]";
  root.addEventListener("input", event => {
    if (event.target.matches(watched)) changed();
  }, { signal });
  root.addEventListener("change", event => {
    const target = event.target;
    if (!target.matches(watched)) return;
    if (target.matches('[data-aic-setting="mode"]')) syncMode();
    if (target.matches('[data-aic-field="type"]')) syncFieldRow(target.closest("[data-aic-item]"));
    changed();
  }, { signal });
  editor.addEventListener("click", event => {
    const button = event.target.closest("button");
    if (!button || button.disabled || readOnly() || !editor.contains(button)) return;
    const container = button.closest("[data-aic-list]");
    const row = button.closest("[data-aic-item]");
    if (button.matches("[data-aic-add]") && container) addRow(container);
    else if (button.matches("[data-aic-move]") && row) move(row, button.dataset.aicMove);
    else if (button.matches("[data-aic-remove]") && row && container) remove(row, container);
    else if (button.matches("[data-aic-test]")) void preview();
  }, { signal });
  apply.addEventListener("click", event => { event.preventDefault(); void save(); }, { signal });
  root.querySelector("[data-block-title-field]")?.addEventListener("keydown", event => {
    if (event.key === "Enter") { event.preventDefault(); void save(); }
  }, { signal });
  if (readOnly()) {
    for (const control of editor.querySelectorAll("[data-aic-field], [data-aic-setting], [data-aic-test]")) control.disabled = true;
  }
  sync();
  editor.dataset.aicBound = "true";
  return () => {
    controller.abort();
    clearTimeout(timer);
    delete editor.dataset.aicBound;
  };
}
