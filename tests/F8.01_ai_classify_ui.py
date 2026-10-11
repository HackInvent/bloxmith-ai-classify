"""FB8: real release UI — mode switch, category and field editors, Apply adding outputs, sample test, EN/FR, responsive."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(Path(__file__).resolve().parent)]
from playwright.sync_api import expect, sync_playwright
from block_test_artifacts import artifact_path
from block_test_packages import install_test_package, surface_payload
from ui_smoke_common import create_project_api, graph_payload, isolated_server, project_editor_url
from blocs.ai_classify.block import AiClassifyBlock
from llm_fixture import llm_fixture

SETTLE_MS = 20_000  # Shared test hosts can be slow; assertions still fail on real defects.
VIEWPORTS = ((1440, 900), (800, 700), (390, 740), (320, 568))
GEOMETRY = """(el, bodySelector) => {
  const r = el.getBoundingClientRect(), body = el.querySelector(bodySelector) || el;
  const actions = [...el.querySelectorAll('[data-close-block-modal], [data-owned-apply]')].filter(a => a.getClientRects().length);
  const boxes = [...el.querySelectorAll('[data-aic-item]')].filter(box => box.getClientRects().length);
  return {
    inside: r.left >= -1 && r.right <= innerWidth + 1 && r.top >= -1 && r.bottom <= innerHeight + 1,
    overflow: el.scrollWidth > el.clientWidth + 2 || body.scrollWidth > body.clientWidth + 2
      || boxes.some(box => box.scrollWidth > box.clientWidth + 2),
    controls: actions.every(a => { const b = a.getBoundingClientRect(); return b.top >= 0 && b.bottom <= innerHeight + 1; }),
    background: getComputedStyle(el).backgroundColor,
    unlabeled: [...el.querySelectorAll('input, select, textarea')].filter(c => !c.labels?.length
      && !c.getAttribute('aria-label') && !c.getAttribute('aria-labelledby')).length,
    small: [...el.querySelectorAll('[data-aic-item] button, [data-aic-add], [data-aic-test]')]
      .filter(b => b.getClientRects().length && b.getBoundingClientRect().height < 32).length,
  };
}"""


def geometry_ok(measured):
    return (measured["inside"] and measured["controls"] and not measured["overflow"] and not measured["unlabeled"]
            and not measured["small"] and measured["background"] not in {"transparent", "rgba(0, 0, 0, 0)"})


def geometry(panel, body=".owned-body"):
    """Opaque panel inside the viewport, reachable actions, labeled controls, no horizontal overflow (polled)."""
    deadline = time.monotonic() + SETTLE_MS / 1000
    measured = panel.evaluate(GEOMETRY, body)
    while not geometry_ok(measured) and time.monotonic() < deadline:
        panel.page.wait_for_timeout(100)
        measured = panel.evaluate(GEOMETRY, body)
    assert geometry_ok(measured), measured


def resize(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_function("([w, h]) => innerWidth === w && innerHeight === h", arg=[width, height], timeout=SETTLE_MS)
    page.evaluate("() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")


class Renders:
    """Track in-flight surface renders: the shell re-renders the inspector after selection and graph changes."""

    def __init__(self, page):
        self.pending, self.last = 0, time.monotonic()
        page.on("request", lambda request: self.track(request, 1))
        page.on("requestfinished", lambda request: self.track(request, -1))
        page.on("requestfailed", lambda request: self.track(request, -1))

    def track(self, request, delta):
        if request.url.endswith(("/inspector-panel", "/modal")):
            self.pending, self.last = max(0, self.pending + delta), time.monotonic()

    def settle(self, page, quiet=1.2, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            page.wait_for_timeout(100)
            if self.pending == 0 and time.monotonic() - self.last >= quiet:
                return
        raise AssertionError("Inspector renders did not settle")


def rows(scope, name):
    return scope.locator(f'[data-aic-list="{name}"] [data-aic-rows] > [data-aic-item]')


def row(scope, name, number):
    return rows(scope, name).nth(number - 1)


def field(item, name):
    return item.locator(f'[data-aic-field="{name}"]')


def setting(scope, key):
    return scope.locator(f'[data-aic-setting="{key}"]')


def mode(scope, value):
    scope.locator(".aic-mode-choice .aic-segment-option", has_text=value).click()
    expect(scope.locator(f'[data-aic-mode-section="{value}"]')).to_be_visible()


def open_modal(page, renders, card, modal):
    """Open the modal once and wait until its editor is bound and no surface render is in flight.

    A second double-click while a slow render is pending would replace the editor under the test.
    """
    expect(card.locator("h3")).to_be_visible(timeout=30_000)
    renders.settle(page)
    bound = modal.locator("[data-aic-editor][data-aic-bound]")
    if not modal.is_visible():
        card.locator("h3").dblclick()
        try:
            expect(bound).to_be_visible(timeout=45_000)
        except AssertionError:
            if not modal.is_visible():
                card.locator("h3").dblclick()
    expect(bound).to_be_visible(timeout=45_000)
    renders.settle(page)


def saving(response):
    return (response.url.endswith("/ui-action") and response.request.method == "POST"
            and "save_settings" in (response.request.post_data or ""))


def preview(scope, sample, outcome, *texts, outputs=None):
    """# - FB8 - the sample test calls the configured endpoint with the draft and shows the decision."""
    result = scope.locator("[data-aic-preview-result]")
    scope.locator("[data-aic-sample]").fill(sample)
    scope.locator("[data-aic-test]").click()
    expect(result).to_have_attribute("data-outcome", outcome)
    if outputs is not None:
        expect(result).to_have_attribute("data-outputs", outputs)
    for text in texts:
        expect(result).to_contain_text(text)


def edit_categories(modal):
    """# - FB7 - FB8 - add, validate, reorder and remove categories; output states and the outputs Apply adds."""
    categories = rows(modal, "categories")
    expect(categories).to_have_count(2)
    expect(row(modal, "categories", 1).locator("[data-aic-output-state]")).to_have_text("Output ready")
    modal.locator('[data-aic-list="categories"] [data-aic-add]').click()
    expect(categories).to_have_count(3)
    third = row(modal, "categories", 3)
    expect(field(third, "name")).to_be_focused()
    expect(modal.locator('[data-aic-list="categories"] [data-aic-count]')).to_have_text("3 / 32")
    field(third, "name").fill("Billing Question")
    error = third.locator("[data-aic-item-error]")
    expect(error).to_contain_text("is not a valid category name")
    expect(field(third, "name")).to_have_attribute("aria-invalid", "true")
    field(third, "name").fill("billing")
    field(third, "description").fill("Invoices, payments and refunds.")
    expect(error).to_be_hidden()
    expect(third.locator("[data-aic-output-state]")).to_have_text("Output added on Apply")
    expect(modal.locator("[data-aic-outputs-status]")).to_have_text("Apply adds the outputs billing.")
    third.locator('[data-aic-move="up"]').click()
    expect(field(row(modal, "categories", 2), "name")).to_have_value("billing")
    expect(row(modal, "categories", 2).locator("[data-aic-number]")).to_have_text("2")
    row(modal, "categories", 3).locator("[data-aic-remove]").click()
    expect(categories).to_have_count(2)
    expect(row(modal, "categories", 1).locator("[data-aic-remove]")).to_be_disabled()
    expect(modal.locator("[data-aic-outputs-status]")).to_contain_text("Outputs no category uses: problem")
    row(modal, "categories", 2).locator('[data-aic-move="up"]').click()
    expect(field(row(modal, "categories", 1), "name")).to_have_value("billing")
    setting(modal, "multi_label").check()


def edit_fields(modal):
    """# - FB4 - FB8 - the field editor: types, enum values, required flag, add and remove."""
    mode(modal, "extract")
    expect(modal.locator('[data-aic-mode-section="classify"]')).to_be_hidden()
    expect(modal.locator("[data-aic-outputs-status]")).to_have_text("In this mode only result and error receive messages.")
    fields = rows(modal, "fields")
    expect(fields).to_have_count(5)
    first = row(modal, "fields", 1)
    expect(first.locator("[data-aic-values-field]")).to_be_hidden()
    expect(field(row(modal, "fields", 4), "values_text")).to_have_value("low, normal, high")
    field(first, "type").select_option("enum")
    expect(first.locator("[data-aic-values-field]")).to_be_visible()
    expect(first.locator("[data-aic-item-error]")).to_contain_text("An enum field needs 1 to 64 allowed values.")
    field(first, "values_text").fill("retail, business")
    expect(first.locator("[data-aic-item-error]")).to_be_hidden()
    modal.locator('[data-aic-list="fields"] [data-aic-add]').click()
    added = row(modal, "fields", 6)
    field(added, "name").fill("vat_number")
    field(added, "required").check()
    field(added, "description").fill("VAT identification number.")
    row(modal, "fields", 2).locator("[data-aic-remove]").click()
    expect(fields).to_have_count(5)
    mode(modal, "summarize")
    setting(modal, "summary_max_chars").fill("50")
    expect(modal.locator('[data-aic-setting-error="summary_max_chars"]')).to_contain_text("from 100 to 20000")
    setting(modal, "summary_max_chars").fill("600")
    expect(modal.locator('[data-aic-setting-error="summary_max_chars"]')).to_be_hidden()
    modal.locator(".aic-segment-option", has_text="paragraph").click()
    setting(modal, "summary_language").select_option("fr")
    mode(modal, "classify")


def main():
    expect.set_options(timeout=SETTLE_MS)
    for origin in ("managed", "linked"):
        with llm_fixture() as api, isolated_server() as server, sync_playwright() as playwright:
            model = install_test_package(server, "ai_classify", origin=origin)
            block = AiClassifyBlock()
            candidate = block.build_node_payload(node_id="ui-case", position={"x": 160, "y": 160})
            candidate["config"].update(api_base_url=api.base_url, model="fixture-llm")
            stale = block.build_node_payload(node_id="ui-stale", position={"x": 160, "y": 460})
            stale["config"].update(api_base_url=api.base_url, model="fixture-llm",
                                   categories=[{"name": "question", "description": ""}, {"name": "refund", "description": ""}])
            for node in (candidate, stale):
                node["block_version"] = model["version"]
            # - FB8 - Every surface declares its assets and a complete EN/FR catalog for its markers.
            for surface in ("modal", "inspector_panel", "node_card"):
                assert "{{" not in surface_payload(server, model, candidate, surface=surface)["html"]
            project = create_project_api(server, document=graph_payload("AI Classify editor", [candidate, stale], []))["project"]
            url = project_editor_url(server.base_url, project["project_id"], workspace_project_id=project["workspace_project_id"])
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                renders = Renders(page)
                page.add_init_script("window.localStorage.setItem('bloxsmith.inspectorPinned','true')")
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                card = page.locator('.canvas-node[data-node-id="ui-case"]')
                expect(card.locator(".node-data-preview")).to_contain_text("classify question · problem fixture-llm")
                card.click(position={"x": 20, "y": 20})
                inspector = page.locator('[data-properties-surface="inspector"][data-node-id="ui-case"]:visible')
                expect(inspector).to_be_visible()
                renders.settle(page)
                modal = page.locator(".aic-modal")
                open_modal(page, renders, card, modal)
                apply = modal.locator("[data-owned-apply]")
                expect(apply).to_be_disabled()
                expect(modal.locator('[data-aic-setting="mode"][value="classify"]')).to_be_checked()
                expect(modal.locator('[data-aic-mode-section="extract"]')).to_be_hidden()
                # Cancel discards a draft.
                field(row(modal, "categories", 1), "description").fill("changed")
                expect(apply).to_be_enabled()
                modal.locator(".owned-footer [data-close-block-modal]").click()
                open_modal(page, renders, card, modal)
                expect(field(row(modal, "categories", 1), "description")).to_have_value(
                    "The sender asks for information, help or an explanation.")
                expect(apply).to_be_disabled()
                edit_categories(modal)
                edit_fields(modal)
                preview(modal, "I have a billing question", "ok", "The message would be sent to billing, question.",
                        "Model fixture-llm · 1 request", outputs="billing question")
                preview(modal, "Good morning", "ok", "No category fits: the message would be sent to unmatched.",
                        outputs="unmatched")
                setting(modal, "model").fill("fixture-429")
                preview(modal, "a question", "error", "The endpoint is rate limiting", "HTTP 429: Rate limit reached")
                setting(modal, "model").fill("fixture-llm")
                setting(modal, "api_key_ref").fill("secret://workspace/llm_key")
                preview(modal, "a question", "invalid", "The editor cannot read the wallet")
                setting(modal, "api_key_ref").fill("")
                with page.expect_response(saving) as saved:
                    apply.click()
                assert not saved.value.json().get("error"), saved.value.json()
                assert saved.value.json()["created_outputs"] == ["billing"], saved.value.json()
                expect(modal.locator("[data-aic-status]")).to_have_text("Settings applied. Outputs added: billing.")
                expect(apply).to_be_disabled()
                expect(row(modal, "categories", 1).locator("[data-aic-output-state]")).to_have_text("Output ready")
                modal.locator(".owned-footer [data-close-block-modal]").click()
                expect(card.locator(".node-data-preview")).to_contain_text("classify billing · question (multi-label)")
                page.reload()
                expect(card.locator(".node-data-preview")).to_contain_text("classify billing · question (multi-label)")
                open_modal(page, renders, card, modal)
                expect(rows(modal, "categories")).to_have_count(2)
                expect(field(row(modal, "categories", 1), "name")).to_have_value("billing")
                expect(field(row(modal, "categories", 1), "description")).to_have_value("Invoices, payments and refunds.")
                expect(setting(modal, "multi_label")).to_be_checked()
                expect(field(row(modal, "fields", 1), "values_text")).to_have_value("retail, business")
                expect(field(row(modal, "fields", 5), "name")).to_have_value("vat_number")
                expect(field(row(modal, "fields", 5), "required")).to_be_checked()
                expect(setting(modal, "summary_max_chars")).to_have_value("600")
                expect(modal.locator("[data-aic-outputs-status]")).to_contain_text("Outputs no category uses: problem")
                for width, height in VIEWPORTS:
                    resize(page, width, height)
                    geometry(modal)
                    page.screenshot(path=artifact_path(f"ai_classify-{origin}-{width}.png"))
                mode(modal, "extract")
                resize(page, 390, 740)
                geometry(modal)
                page.screenshot(path=artifact_path(f"ai_classify-{origin}-extract-390.png"))
                resize(page, 1440, 900)
                mode(modal, "classify")
                # Keyboard: add then remove a category without the mouse.
                modal.locator('[data-aic-list="categories"] [data-aic-add]').focus()
                page.keyboard.press("Enter")
                expect(rows(modal, "categories")).to_have_count(3)
                expect(field(row(modal, "categories", 3), "name")).to_be_focused()
                row(modal, "categories", 3).locator("[data-aic-remove]").focus()
                page.keyboard.press("Enter")
                expect(rows(modal, "categories")).to_have_count(2)
                modal.locator(".owned-footer [data-close-block-modal]").click()
                # - FB7 - stored categories without outputs: the card warns, Apply adds the outputs at once.
                stale_card = page.locator('.canvas-node[data-node-id="ui-stale"]')
                expect(stale_card.locator(".node-data-preview")).to_contain_text("Invalid settings")
                open_modal(page, renders, stale_card, modal)
                expect(modal.locator("[data-aic-outputs-status]")).to_contain_text("Apply adds the outputs refund.")
                expect(apply).to_be_enabled()
                with page.expect_response(saving) as saved:
                    apply.click()
                assert saved.value.json()["created_outputs"] == ["refund"], saved.value.json()
                modal.locator(".owned-footer [data-close-block-modal]").click()
                expect(stale_card.locator(".node-data-preview")).to_contain_text("classify question · refund")
                # Inspector: same editor, Ports tab with the added output, sample test and Apply from the toolbar.
                card.click(position={"x": 20, "y": 20})
                expect(inspector).to_be_visible()
                renders.settle(page)
                inspector.locator('[data-inspector-tab="ports"]').click()
                expect(inspector.locator('.port-name-input[data-direction="output"]')).to_have_count(6)
                expect(inspector.locator('[data-inspector-panel-tab="ports"]')).to_contain_text("Apply in the General tab")
                inspector.locator('[data-inspector-tab="general"]').click()
                renders.settle(page)
                expect(rows(inspector, "categories")).to_have_count(2)
                assert inspector.evaluate("el => el.scrollWidth <= el.clientWidth + 2")
                preview(inspector, "A billing problem", "ok", "The message would be sent to billing.", outputs="billing")
                field(row(inspector, "categories", 2), "description").fill("Requests for information.")
                with page.expect_response(saving) as saved:
                    inspector.locator("[data-owned-apply]").click()
                assert not saved.value.json().get("error"), saved.value.json()
                page.screenshot(path=artifact_path(f"ai_classify-{origin}-inspector.png"))
                for width, height in ((390, 740), (320, 568)):
                    resize(page, width, height)
                    page.screenshot(path=artifact_path(f"ai_classify-{origin}-inspector-{width}.png"))
                resize(page, 1440, 900)
                # French: translated chrome, live validation, preview and card follow the language.
                page.goto(server.base_url + "/")
                page.locator("#homeApplicationSettingsButton").click()
                page.locator("#applicationLanguageSelect").select_option("fr")
                page.wait_for_function("window.CWMessages.getLanguage() === 'fr'")
                page.goto(url)
                expect(card.locator(".node-data-preview")).to_contain_text("(multi-étiquette)")
                open_modal(page, renders, card, modal)
                expect(modal.locator('[data-aic-list="categories"] [data-aic-add]')).to_have_text("Ajouter une catégorie")
                expect(modal.locator("[data-aic-test]")).to_have_text("Tester avec le modèle")
                expect(modal.locator("[data-aic-advanced] > summary")).to_have_text("Réglages avancés")
                expect(modal.locator(".panel-kicker")).to_have_text("Analyse de texte par IA")
                field(row(modal, "categories", 2), "name").fill("question!")
                expect(row(modal, "categories", 2).locator("[data-aic-item-error]")).to_contain_text(
                    "n’est pas un nom de catégorie valide")
                field(row(modal, "categories", 2), "name").fill("question")
                preview(modal, "A billing question", "ok", "Le message serait envoyé vers billing, question.",
                        "Modèle fixture-llm · 1 requête", outputs="billing question")
                mode(modal, "summarize")
                preview(modal, "Une première phrase courte. Puis une seconde phrase.", "ok", "Résumé (")
                for width, height in ((1440, 900), (390, 740)):
                    resize(page, width, height)
                    geometry(modal)
                    page.screenshot(path=artifact_path(f"ai_classify-{origin}-fr-{width}.png"))
                resize(page, 1440, 900)
                modal.locator(".owned-footer [data-close-block-modal]").click()
                assert not errors, errors
                assert page.evaluate("!window.CWBlockUiBlocks?.ai_classify")
            finally:
                browser.close()
        print(f"[ok] ai_classify {origin} UI", flush=True)


if __name__ == "__main__":
    main()
