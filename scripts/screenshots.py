"""Capture dashboard screenshots for the README.

Needs a running dashboard (streamlit run app/dashboard.py) and
`pip install playwright`. Uses the installed Edge or Chrome, so there is no
browser download.

    python scripts/screenshots.py [--url http://localhost:8501] [--browser msedge]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
TABS = {  # tab label -> file name
    "Thesis": "thesis",
    "Active risk": "active_risk",
    "Risk": "risk",
    "Correlations & regimes": "correlations",
    "Trade proposal": "trade_proposal",
    "Stress tests": "stress",
    "Monte Carlo": "monte_carlo",
}


def wait_idle(page, timeout=180):
    """Wait until Streamlit has finished running the script and charts have drawn."""
    end = time.time() + timeout
    time.sleep(1.5)
    while time.time() < end:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        spinners = page.locator('[data-testid="stSpinner"]').count() > 0
        if not running and not spinners:
            break
        time.sleep(1)
    time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--browser", default="msedge", help="msedge or chrome")
    ap.add_argument("--height", type=int, default=1900)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": args.height},
                                color_scheme="light", device_scale_factor=1)
        page.goto(args.url)
        page.get_by_text("Investment thesis").first.wait_for(timeout=240_000)
        wait_idle(page)
        # hide the sidebar so the charts get the full width
        page.add_style_tag(content='[data-testid="stSidebar"]{display:none !important}')
        time.sleep(1)
        for label, name in TABS.items():
            page.get_by_role("tab", name=label, exact=True).click()
            wait_idle(page)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
