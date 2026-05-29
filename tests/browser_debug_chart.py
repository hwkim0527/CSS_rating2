from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 1200})
    msgs = []
    page.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
    page.on("pageerror", lambda e: msgs.append(f"[ERR] {e}"))
    page.goto("http://127.0.0.1:8766/compare", wait_until="networkidle")
    info = page.evaluate("""() => {
        const c = document.getElementById('auc-chart');
        if (!c) return {found: false};
        const rect = c.getBoundingClientRect();
        const ctx = c.getContext('2d');
        const data = ctx.getImageData(rect.width/2, rect.height/2, 1, 1).data;
        return {
            found: true,
            width: c.width, height: c.height,
            cssW: rect.width, cssH: rect.height,
            centerPixel: Array.from(data),
            metricsKeys: window.METRICS ? Object.keys(window.METRICS) : null,
            modelKeys: window.METRICS && window.METRICS.models ? Object.keys(window.METRICS.models) : null,
        };
    }""")
    print("INFO:", info)
    print("MSGS:")
    for m in msgs:
        print(" ", m)
    page.locator('canvas#auc-chart').screenshot(path="C:/tmp/css_canvas_only.png")
    browser.close()
