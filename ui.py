"""Package-owned editor surfaces, sharing only the framework's public rendering helpers."""

from html import escape
import json
import re
import secrets

from bloxsmith_app.block_api import render_inspector_template
from . import logic

HELP = {
    "mode": "classify: route the message to its category. sentiment: route it by tone and add a score. "
            "extract: read declared fields into JSON. summarize: write a summary. Every mode publishes result.",
    "multi_label": "Off: at most one category. On: every category that applies; a copy goes to each of their outputs.",
    "categories": "Each category sends the unchanged message to the output of the same name. Messages that fit none go "
                  "to unmatched. Apply adds the outputs of new categories.",
    "sentiment_categories": "The model picks exactly one label and a score from 0 (most negative) to 1 (most positive). "
                            "The message goes to the output of that label.",
    "fields": "Missing values are null, never guessed. Values of the wrong type are null and listed in errors.",
    "category_name": "name: lowercase letters, digits and _ (at most 20). It is also the output name.",
    "description": "description: what belongs in this category, in plain words; the model reads it.",
    "field_values": "values: allowed values of an enum field, separated by commas.",
    "field_description": "description: what the field contains and where to find it; the model reads it.",
    "sample": "Calls the configured endpoint once with the settings being edited, including unapplied changes. "
              "Nothing is saved, routed or published. Endpoints that need api_key_ref cannot be tested here.",
}
FIELD_TYPE_OPTIONS = logic.FIELD_TYPES


def replace_config(result, node):
    """Replace edited settings exactly; generic graph patches otherwise deep-merge objects."""
    patch = result.get("node_patch") if isinstance(result, dict) else None
    if not isinstance(patch, dict) or not isinstance(patch.get("config"), dict):
        return result
    settings = {**(node.get("config") or {}), **patch.pop("config")}
    result.setdefault("graph_operations", []).append({"op": "update_node_config",
        "node_id": str(node.get("id") or ""), "config": settings, "replace": True})
    return result


def _text(block, key, fallback, tag="span", attrs="", params=None):
    """Catalog-owned text with its marker; never used for user content."""
    name = f"block.{block.kind}.{key}"
    marker = f' data-i18n-params="{escape(json.dumps(params), quote=True)}"' if params else ""
    return (f'<{tag}{attrs} data-i18n="{name}"{marker}>'
            f'{escape(block.translate(name, params, fallback=fallback))}</{tag}>')


def _attr(block, name, key, fallback):
    """Translated attribute plus its marker, e.g. aria-label."""
    full = f"block.{block.kind}.{key}"
    return f' {name}="{escape(block.translate(full, fallback=fallback), quote=True)}" data-i18n-{name}="{full}"'


def _hint(block, key, element_id, fallback=None, tag="p"):
    return _text(block, f"help_{key}", fallback or HELP[key], tag, f' id="{element_id}" class="field-hint"')


def _options(values, selected):
    return "".join(f'<option value="{escape(value, quote=True)}"{" selected" if value == selected else ""}>'
                   f'{escape(value)}</option>' for value in values)


def _actions(block, scope):
    up = _attr(block, "aria-label", f"{scope}_move_up", "Move up") + _attr(block, "title", f"{scope}_move_up", "Move up")
    down = (_attr(block, "aria-label", f"{scope}_move_down", "Move down")
            + _attr(block, "title", f"{scope}_move_down", "Move down"))
    return ('<span class="aic-actions">'
            f'<button type="button" class="ghost-btn aic-icon" data-aic-move="up" data-block-mutation-control{up}>'
            '<span aria-hidden="true">↑</span></button>'
            f'<button type="button" class="ghost-btn aic-icon" data-aic-move="down" data-block-mutation-control{down}>'
            '<span aria-hidden="true">↓</span></button>'
            '<button type="button" class="ghost-btn aic-remove" data-aic-remove data-block-mutation-control>'
            f'{_text(block, "remove", "Remove")}</button></span>')


def category_row(block, prefix, item, number, label_key, label):
    """One category: its name (also its output name) and its description."""
    item = item if isinstance(item, dict) else {}
    name = item.get("name") if isinstance(item.get("name"), str) else ""
    description = item.get("description") if isinstance(item.get("description"), str) else ""
    return (
        '<li class="aic-row" data-aic-item>'
        '<div class="aic-row-head">'
        f'<span class="aic-row-title">{_text(block, label_key, label)} <span data-aic-number>{number}</span></span>'
        '<span class="aic-output-state" data-aic-output-state></span>'
        f'{_actions(block, "item")}</div>'
        '<div class="aic-row-fields aic-category-fields">'
        '<label class="aic-field"><span class="aic-key">name</span>'
        f'<input type="text" data-aic-field="name" value="{escape(name, quote=True)}" maxlength="40" spellcheck="false" '
        f'autocomplete="off" placeholder="billing_question" aria-describedby="{prefix}-help-category-name"></label>'
        '<label class="aic-field"><span class="aic-key">description</span>'
        f'<textarea data-aic-field="description" rows="2" maxlength="{logic.MAX_DESCRIPTION}" '
        f'aria-describedby="{prefix}-help-description">{escape(description)}</textarea></label>'
        '</div><p class="aic-error" data-aic-item-error hidden></p></li>'
    )


def field_row(block, prefix, item, number):
    """One extracted field: name, type, enum values, required flag and description."""
    item = item if isinstance(item, dict) else {}
    name = item.get("name") if isinstance(item.get("name"), str) else ""
    kind = item.get("type") if item.get("type") in FIELD_TYPE_OPTIONS else "string"
    description = item.get("description") if isinstance(item.get("description"), str) else ""
    checked = " checked" if item.get("required") is True else ""
    values = logic.values_text(item.get("values")) if kind == "enum" else ""
    hidden = "" if kind == "enum" else " hidden"
    return (
        '<li class="aic-row" data-aic-item>'
        '<div class="aic-row-head">'
        f'<span class="aic-row-title">{_text(block, "field_label", "Field")} <span data-aic-number>{number}</span></span>'
        f'{_actions(block, "item")}</div>'
        '<div class="aic-row-fields aic-field-fields">'
        '<label class="aic-field"><span class="aic-key">name</span>'
        f'<input type="text" data-aic-field="name" value="{escape(name, quote=True)}" maxlength="80" spellcheck="false" '
        'autocomplete="off" placeholder="invoice_number"></label>'
        '<label class="aic-field"><span class="aic-key">type</span>'
        f'<select data-aic-field="type">{_options(FIELD_TYPE_OPTIONS, kind)}</select></label>'
        f'<label class="aic-field aic-values" data-aic-values-field{hidden}><span class="aic-key">values</span>'
        f'<input type="text" data-aic-field="values_text" value="{escape(values, quote=True)}" spellcheck="false" '
        f'autocomplete="off" placeholder="low, normal, high" aria-describedby="{prefix}-help-values"></label>'
        '<label class="aic-check"><input type="checkbox" data-aic-field="required"' + checked + '>'
        '<span class="aic-key">required</span></label>'
        '<label class="aic-field aic-wide"><span class="aic-key">description</span>'
        f'<textarea data-aic-field="description" rows="2" maxlength="{logic.MAX_DESCRIPTION}" '
        f'aria-describedby="{prefix}-help-field-description">{escape(description)}</textarea></label>'
        '</div><p class="aic-error" data-aic-item-error hidden></p></li>'
    )


def _list(block, prefix, list_name, rows, template, add_key, add_label, minimum, maximum):
    count = f"{len(rows)} / {maximum}"
    return (
        f'<div class="aic-list" data-aic-list="{list_name}" data-min="{minimum}" data-max="{maximum}">'
        f'<div class="aic-list-head"><h3 class="aic-key">{list_name}</h3>'
        f'<span class="field-hint" data-aic-count>{count}</span></div>'
        f'{_hint(block, list_name, f"{prefix}-help-{list_name}")}'
        f'<ol class="aic-rows" data-aic-rows>{"".join(rows)}</ol>'
        f'<template data-aic-template>{template}</template>'
        '<div class="aic-add-line"><button type="button" class="ghost-btn aic-add" data-aic-add data-block-mutation-control>'
        f'{_text(block, add_key, add_label)}</button></div>'
        '<p class="aic-error" data-aic-list-error role="status" hidden></p></div>'
    )


def _editable_list(raw, fallback, limit):
    return raw[:limit] if isinstance(raw, list) and raw else fallback


def _setting(block, prefix, spec, config):
    """One scalar setting from fields.json, with its hint and its error slot."""
    key, kind = spec["key"], spec.get("type", "text")
    value = config.get(key)
    ident = f"{prefix}-{key}"
    hint = _hint(block, key, f"{ident}-hint", spec["help"])
    error = f'<p class="aic-error" data-aic-setting-error="{key}" hidden></p>'
    common = f'id="{ident}" data-aic-setting="{key}" aria-describedby="{ident}-hint"'
    if kind == "boolean":
        checked = " checked" if value is True else ""
        return (f'<div class="field-group aic-check-group"><input {common} type="checkbox"{checked}>'
                f'<label for="{ident}">{key}</label>{hint}{error}</div>')
    if kind == "radio":
        chosen = value if value in spec["options"] else spec["options"][0]
        options = "".join(
            f'<label class="aic-segment-option"><input type="radio" name="{ident}" value="{option}" data-aic-setting="{key}"'
            f'{" checked" if option == chosen else ""} aria-describedby="{ident}-hint"><span>{option}</span></label>'
            for option in spec["options"])
        return (f'<fieldset class="aic-choice"><legend class="aic-key">{key}</legend>'
                f'<div class="aic-segment">{options}</div>{hint}{error}</fieldset>')
    if kind == "select":
        chosen = value if value in spec["options"] else spec["options"][0]
        control = f'<select {common}>{_options(spec["options"], chosen)}</select>'
    elif kind in {"integer", "number"}:
        shown = "" if value is None else str(value)
        step = spec.get("step", "1")
        nullable = ' data-aic-nullable="true"' if spec.get("nullable") else ""
        control = (f'<input {common} type="number" inputmode="decimal" min="{spec["min"]}" max="{spec["max"]}" '
                   f'step="{step}" data-aic-type="{kind}"{nullable} value="{escape(shown, quote=True)}">')
    elif kind == "textarea":
        placeholder = escape(spec.get("placeholder", ""), quote=True)
        control = (f'<textarea {common} rows="3" maxlength="{logic.MAX_INSTRUCTIONS}" placeholder="{placeholder}">'
                   f'{escape(value if isinstance(value, str) else "")}</textarea>')
    else:
        placeholder = escape(spec.get("placeholder", ""), quote=True)
        shown = value if isinstance(value, str) else ""
        control = (f'<input {common} type="text" autocomplete="off" spellcheck="false" placeholder="{placeholder}" '
                   f'value="{escape(shown, quote=True)}">')
    return f'<div class="field-group"><label for="{ident}">{key}</label>{control}{hint}{error}</div>'


def _settings(block, prefix, config, section):
    specs = json.loads((block.directory / "fields.json").read_text(encoding="utf-8"))
    return "".join(_setting(block, prefix, spec, config) for spec in specs if spec["section"] == section)


def editor_html(block, node, surface):
    """The complete editor shared by the modal and the inspector."""
    stored = node.get("config") if isinstance(node.get("config"), dict) else {}
    config = logic.settings(stored)
    prefix = escape(re.sub(r"[^A-Za-z0-9_-]", "-", f"{block.kind}-{node.get('id', '')}-{surface}"), quote=True)
    mode = config.get("mode") if config.get("mode") in logic.MODES else "classify"
    pairs = logic.node_pairs(node) or []
    outputs = [name for ident, name in pairs if ident not in logic.FIXED_PORTS]
    fixed = [name for ident, name in pairs if ident in logic.FIXED_PORTS]
    initial = validation({}, stored, logic.node_pairs(node))
    segment = "".join(
        f'<label class="aic-segment-option"><input type="radio" name="{prefix}-mode" value="{value}" data-aic-setting="mode"'
        f'{" checked" if value == mode else ""} aria-describedby="{prefix}-help-mode"><span>{value}</span></label>'
        for value in logic.MODES)
    lists = {}
    for list_name, defaults, label_key, label in (("categories", logic.DEFAULT_CATEGORIES, "category_label", "Category"),
                                                  ("sentiment_categories", logic.DEFAULT_SENTIMENT, "label_label", "Label")):
        items = _editable_list(config.get(list_name), defaults, logic.MAX_CATEGORIES)
        rows = [category_row(block, prefix, item, number, label_key, label) for number, item in enumerate(items, start=1)]
        add_key, add_label = ("add_category", "Add a category") if list_name == "categories" else ("add_label", "Add a label")
        lists[list_name] = _list(block, prefix, list_name, rows, category_row(block, prefix, {}, 1, label_key, label),
                                 add_key, add_label, logic.MIN_CATEGORIES, logic.MAX_CATEGORIES)
    field_items = _editable_list(config.get("fields"), logic.DEFAULT_FIELDS, logic.MAX_FIELDS)
    field_rows = [field_row(block, prefix, item, number) for number, item in enumerate(field_items, start=1)]
    lists["fields"] = _list(block, prefix, "fields", field_rows, field_row(block, prefix, {}, 1), "add_field",
                            "Add a field", 1, logic.MAX_FIELDS)
    multi = " checked" if config.get("multi_label") is True else ""

    def section(name, body):
        hidden = "" if name == mode else " hidden"
        return f'<section class="aic-mode" data-aic-mode-section="{name}"{hidden}>{body}</section>'

    routing_help = (f'<div class="aic-help" data-aic-routing-help{"" if mode in logic.ROUTING_MODES else " hidden"}>'
                    f'{_hint(block, "category_name", f"{prefix}-help-category-name")}'
                    f'{_hint(block, "description", f"{prefix}-help-description")}</div>')
    sample_attrs = (f'id="{prefix}-sample" data-aic-sample rows="4" maxlength="{logic.MAX_SAMPLE_CHARS}" '
                    f'aria-describedby="{prefix}-help-sample"')
    connection_attrs = f' id="{prefix}-connection"'
    preview_attrs = f' id="{prefix}-preview"'
    return (
        f'<div class="aic-editor" data-aic-editor data-surface="{escape(surface, quote=True)}" '
        f'data-aic-outputs="{escape(json.dumps(outputs), quote=True)}" data-aic-fixed="{escape(json.dumps(fixed), quote=True)}" '
        f'data-aic-initial="{escape(json.dumps(initial), quote=True)}">'
        f'<fieldset class="aic-choice aic-mode-choice"><legend class="aic-key">mode</legend>'
        f'<div class="aic-segment">{segment}</div>{_hint(block, "mode", f"{prefix}-help-mode")}</fieldset>'
        + section("classify",
                  '<div class="field-group aic-check-group">'
                  f'<input id="{prefix}-multi_label" type="checkbox" data-aic-setting="multi_label"{multi} '
                  f'aria-describedby="{prefix}-help-multi_label"><label for="{prefix}-multi_label">multi_label</label>'
                  f'{_hint(block, "multi_label", f"{prefix}-help-multi_label")}'
                  '<p class="aic-error" data-aic-setting-error="multi_label" hidden></p></div>'
                  + lists["categories"])
        + section("sentiment", lists["sentiment_categories"])
        + routing_help
        + section("extract", lists["fields"] + f'<div class="aic-help">{_hint(block, "field_values", f"{prefix}-help-values")}'
                  f'{_hint(block, "field_description", f"{prefix}-help-field-description")}</div>')
        + section("summarize", f'<div class="owned-fields">{_settings(block, prefix, config, "summarize")}</div>')
        + '<p class="field-hint aic-outputs" data-aic-outputs-status role="status" aria-live="polite"></p>'
        f'<section class="aic-connection" aria-labelledby="{prefix}-connection">'
        f'{_text(block, "connection_title", "Model endpoint", "h3", connection_attrs)}'
        f'<div class="owned-fields">{_settings(block, prefix, config, "connection")}</div></section>'
        '<details class="owned-advanced" data-aic-advanced>'
        f'{_text(block, "advanced", "Advanced settings", "summary")}'
        f'<div class="owned-fields">{_settings(block, prefix, config, "advanced")}</div></details>'
        '<p class="aic-error aic-group-error" data-aic-group-error role="status" aria-live="polite" hidden></p>'
        f'<section class="aic-preview" aria-labelledby="{prefix}-preview">'
        f'{_text(block, "preview_title", "Test with sample text", "h3", preview_attrs)}'
        f'<div class="field-group"><label for="{prefix}-sample">{_text(block, "sample", "Sample text")}</label>'
        f'<textarea {sample_attrs}></textarea>{_hint(block, "sample", f"{prefix}-help-sample")}</div>'
        '<div class="aic-preview-actions"><button type="button" class="ghost-btn" data-aic-test>'
        f'{_text(block, "test", "Test with the model")}</button></div>'
        '<div class="aic-preview-result" data-aic-preview-result role="status" aria-live="polite"></div>'
        '</section></div>'
    )


def validation(raw_draft, stored, outputs):
    """Settings problems of a draft plus the outputs Apply would add and the unused ones (no network)."""
    try:
        draft = logic.settings({**logic.thaw(stored or {}), **logic.editor_settings(raw_draft)})
    except (TypeError, ValueError):
        draft = logic.settings(stored or {})
    errors = [error.describe() for error in logic.problems(draft)]
    report = {"missing": [], "unused": [], "errors": []}
    if outputs is not None and draft.get("mode") in logic.MODES:
        report = logic.output_report(draft, outputs)
        errors += [error.describe() for error in report["errors"]]
    return {"valid": not errors, "errors": errors, "missing_outputs": report["missing"],
            "unused_outputs": report["unused"]}


def _with_editor(template, render, editor):
    """Insert the editor after generic placeholder rendering, through an unguessable marker."""
    marker = f"<!--aic-editor-{secrets.token_hex(8)}-->"
    return render(template.replace("{{ editor }}", marker)).replace(marker, editor)


def modal(block, node, payload):
    """Use the generic draft host; refreshing runtime must not replace pending settings."""
    template = (block.directory / "block_modal.html").read_text(encoding="utf-8")
    html = _with_editor(template, lambda text: block._render_generic_modal_template(
        template=text, node=node, payload=payload or {}), editor_html(block, node, "modal"))
    return {"html": html, "context": {"node_id": str(node.get("id", "")), "node_kind": block.kind}}


def inspector(block, node, payload):
    """Render the same editor with the standard Ports tab and the sticky Apply control."""
    template = (block.directory / "inspector_panel.html").read_text(encoding="utf-8")
    html = _with_editor(template, lambda text: render_inspector_template(
        template=text, node={**node, "kind": block.kind, "type": block.kind}, payload=payload),
        editor_html(block, node, "inspector"))
    return {"html": html, "context": {"node_id": str(node.get("id", "")), "full_panel": True}}


def card_detail(block, node):
    """Mode and what it routes or produces; never the endpoint, the wallet reference or message content."""
    try:
        config = logic.configuration(node.get("config") or {})
        pairs = logic.node_pairs(node)
        report = logic.output_report(config, pairs) if pairs is not None else {"errors": [], "missing": []}
        if report["errors"] or report["missing"]:
            raise report["errors"][0] if report["errors"] else logic.missing_output_errors(config, report["missing"])[0]
    except logic.SettingsError:
        return _text(block, "card_invalid", "Invalid settings: open the block to fix them.", "span", ' class="aic-card-invalid"')
    mode = config["mode"]
    if mode in logic.ROUTING_MODES:
        names = [item["name"] for item in logic.active_categories(config)]
        shown = " · ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
        detail = f'<span class="aic-card-mode">{mode}</span> <span class="aic-card-names">{escape(shown)}</span>'
        if mode == "classify" and config["multi_label"]:
            detail += " " + _text(block, "card_multi", "(multi-label)", "span", ' class="aic-card-extra"')
    elif mode == "extract":
        detail = (f'<span class="aic-card-mode">{mode}</span> '
                  + _text(block, "card_fields", "{count} fields", "span", ' class="aic-card-extra"',
                          {"count": len(config["fields"])}))
    else:
        detail = (f'<span class="aic-card-mode">{mode}</span> '
                  + _text(block, "card_summary", "{style}, at most {max} characters", "span", ' class="aic-card-extra"',
                          {"style": config["summary_style"], "max": config["summary_max_chars"]}))
    return detail + f' <span class="aic-card-model">{escape(config["model"])}</span>'


def card(block, node):
    """Mode summary and model; single-pass substitution so content is never re-substituted."""
    values = {"title": escape(str(node.get("title") or block.default_title())),
              "purpose": escape(block.description(), quote=True), "detail": card_detail(block, node)}
    template = (block.directory / "node_card.html").read_text(encoding="utf-8")
    html = re.sub(r"__(title|purpose|detail)__", lambda match: values[match.group(1)], template)
    return {"html": html, "context": {"node_id": str(node.get("id") or ""), "node_classes": [block.kind + "-node"]}}
