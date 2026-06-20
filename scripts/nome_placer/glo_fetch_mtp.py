"""Enumerate + download BLM GLO Master Title Plat + supplemental sheets for a
Kateel River township, full-resolution. GLO serves each plat through a LizardTech/
OpenSeadragon viewer that auto-converts the JP2 to a plain JPG at
/ConvertedImages/<name>.JPG, which downloads directly. We read each LSR document's
viewer imageFile, then pull the JPG.

  python3 scripts/nome_placer/glo_fetch_mtp.py 11 S 34 W   # T11S R34W, Kateel River (m=44)
Writes data/raw/glo_mtp/nome/<NAME>.jpg + appends to _manifest.json
"""
import asyncio, json, sys, urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path("/home/sky/src/learning/ai-minerals/data/raw/glo_mtp/nome")
HDR = {"User-Agent": "Mozilla/5.0 Chrome/120", "Referer": "https://glorecords.blm.gov/details/lsr/default.aspx"}

async def fetch_township(twp, td, rng, rd):
    OUT.mkdir(parents=True, exist_ok=True)
    results = (f"https://glorecords.blm.gov/results/default.aspx?searchCriteria="
               f"type=map|st=AK|twp_nr={twp}|twp_dir={td}|rng_nr={rng}|rng_dir={rd}|m=44#resultsTabIndex=2")
    got = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(user_agent=HDR["User-Agent"])
        await pg.goto(results, wait_until="networkidle", timeout=60000); await pg.wait_for_timeout(3500)
        lsr = await pg.evaluate(r"""() => [...new Set([...document.querySelectorAll('a')].map(a=>a.href)
            .filter(h=>/details\/lsr\/default/.test(h)))]""")
        print(f"  T{twp}{td} R{rng}{rd}: {len(lsr)} LSR docs")
        for href in lsr:
            await pg.goto(href, wait_until="networkidle", timeout=60000); await pg.wait_for_timeout(1500)
            vsrc = await pg.evaluate(r"""() => { const f=[...document.querySelectorAll('iframe')].map(x=>x.src)
                .find(s=>/lizardtech/.test(s)); return f||null; }""")
            label = await pg.evaluate(r"""() => (document.querySelector('h1,.documentTitle,#docTitle')||{}).innerText||document.title""")
            if not vsrc: 
                print("    (no viewer on", href[-30:], ")"); continue
            # imageFile=AK-AK\NAME.JP2  -> ConvertedImages/NAME.JPG
            import re
            m = re.search(r"imageFile=([^&]+)", vsrc)
            imgf = urllib.request.unquote(m.group(1)).replace("\\","/").split("/")[-1]  # NAME.JP2
            name = imgf.rsplit(".",1)[0]
            jpg = f"https://glorecords.blm.gov/ConvertedImages/{name}.JPG"
            dest = OUT / f"{name}.jpg"
            if not dest.exists():
                try:
                    req = urllib.request.Request(jpg, headers=HDR)
                    data = urllib.request.urlopen(req, timeout=90).read()
                    dest.write_bytes(data)
                    print(f"    downloaded {name}.jpg ({len(data)//1024} KB)  [{(label or '').strip()[:30]}]")
                except Exception as e:
                    print(f"    FAIL {name}: {str(e)[:50]}"); continue
            got.append({"name": name, "twp": f"{twp}{td}", "rng": f"{rng}{rd}", "label": (label or "").strip()[:40], "jpg": jpg})
        await b.close()
    man = OUT / "_manifest.json"
    cur = json.loads(man.read_text()) if man.exists() else []
    names = {g["name"] for g in cur}
    cur += [g for g in got if g["name"] not in names]
    man.write_text(json.dumps(cur, indent=1))
    return got

if __name__ == "__main__":
    twp, td, rng, rd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    asyncio.run(fetch_township(twp, td, rng, rd))
