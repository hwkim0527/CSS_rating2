"""Browser smoke test against running uvicorn (default port 8766)."""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8766"


def main() -> int:
    errors: list[str] = []
    console_msgs: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.on("console", lambda msg: console_msgs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))

        # 1) Home
        page.goto(f"{BASE}/", wait_until="networkidle")
        page.screenshot(path="C:/tmp/css_index.png", full_page=True)
        title = page.title()
        assert "신용평가" in title, f"Bad title: {title}"
        assert page.locator('form#score-form').count() == 1, "form missing"
        assert page.locator('input[name="loan_amnt"]').count() == 1
        # Verify int_rate field is GONE
        assert page.locator('input[name="int_rate"]').count() == 0, "int_rate should be removed"
        print("[1] index page: OK (form rendered, int_rate removed)")

        # 2) Submit form
        page.locator('button[type="submit"]').click()
        page.wait_for_selector('#result:not(.hidden)', timeout=10000)
        page.screenshot(path="C:/tmp/css_result.png", full_page=True)
        score = page.locator('#r-score').inner_text()
        prob = page.locator('#r-prob').inner_text()
        grade = page.locator('#r-grade').inner_text()
        model = page.locator('#r-model').inner_text()
        assert score and score != "—", f"score empty: {score}"
        assert prob and "%" in prob, f"prob bad: {prob}"
        assert "등급" in grade, f"grade bad: {grade}"
        assert "XGBoost" in model, f"model bad: {model}"
        print(f"[2] form submit: score={score} prob={prob} grade={grade} model={model}")

        # 3) Compare page
        page.goto(f"{BASE}/compare", wait_until="networkidle")
        page.screenshot(path="C:/tmp/css_compare.png", full_page=True)
        assert page.locator('table.metrics-table').count() >= 1
        assert page.locator('canvas#auc-chart').count() == 1
        # Verify KS row is visible & shows positive improvement
        body_html = page.content()
        assert "KS" in body_html
        assert "✓ 달성" in body_html, "expected at least one ✓ 달성 badge"
        print("[3] compare page: OK (table + canvas + meets-target badge)")

        # 4) Console errors
        crit = [m for m in console_msgs if m.startswith("[error]") or m.startswith("[warning]")]
        if errors or crit:
            print("[4] console issues:")
            for e in errors + crit:
                print(" ", e)
        else:
            print("[4] no console errors / page errors")

        browser.close()
    print("\nALL BROWSER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
