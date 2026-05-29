"""Headless screenshot of the live goldbug map for portfolio hero use.

Loads https://www.johnsondevco.com/goldbug/goldbug_map.html in a
1600x1000 viewport, waits for tiles + GeoJSON to settle, optionally
clicks the parcel navigator to surface the top-ranked parcel popup,
and saves a PNG.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = "https://www.johnsondevco.com/goldbug/goldbug_map.html"
OUT = Path("data/derived/portfolio_charts/goldbug_screenshot.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    async with async_playwright() as p:
        # chromium-headless-shell is what `playwright install chromium`
        # gave us. Use channel-default; playwright resolves it.
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=2,  # high-DPI for portfolio use
        )
        page = await context.new_page()
        print(f"loading {URL} ...")
        await page.goto(URL, wait_until="networkidle", timeout=60_000)
        # Let map tiles + GeoJSON layers actually paint
        await page.wait_for_timeout(8000)

        # Try to click the navigator "Next" button so a parcel popup is open
        # in the screenshot. This gives a much more compelling hero shot than
        # a bare map. Best-effort; if it fails, still take the screenshot.
        try:
            await page.click("button:has-text('Next')", timeout=5000)
            await page.wait_for_timeout(2500)
            print("clicked Next → popup should be open")
        except Exception as exc:  # noqa: BLE001
            print(f"navigator click skipped: {exc}")

        await page.screenshot(path=str(OUT), full_page=False)
        print(f"wrote {OUT}")

        await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001
        print(f"capture failed: {exc}", file=sys.stderr)
        sys.exit(1)
