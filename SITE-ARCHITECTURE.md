# Site architecture + deploy (handoff, 2026-06-01)

How `johnsondevco.com/ai-minerals/` and its siblings are built and deployed, and
the things that will break it if you forget them. Read this before changing the
site or any deploy script.

## TL;DR — one owner per path

| URL | Owned + deployed by | Contents |
|---|---|---|
| `/ai-minerals/` | **this repo (ai-minerals)** | front-door `index.qmd` + the regional chapters + the deep-review notebooks |
| `/bearcub/` | `~/src/learning/bearcub` | Bear Cub chapter + its deep notebooks |
| `/goldbug/` | `~/src/learning/gldbg` | the live goldbug tool (Cloud-Run/aliased) |

Each repo renders and deploys **only its own path**, with its own deps, data, and
leak guard. There is **no cross-repo include and no shared/aggregating build.**
An earlier "portfolio umbrella" repo tried to pull all chapters into one site via
`{{< include >}}`; it caused deploy collisions and a private-data leak and is
**decommissioned** (its deploy script refuses to run).

## How `/ai-minerals/` is built

- One Quarto site, `_quarto.yml`. Render list = `index.qmd` (front door, FIRST) +
  the chapter qmds (`regional`, `reproductions`, `cross_region`, `drill_planning`,
  `posts`) + the deep notebooks (`notebooks/<region>/*.qmd`, which land at
  `/ai-minerals/notebooks/<region>/<page>.html`).
- `index.qmd` is the intro page: locator map + narrative + chapter cards. It was
  moved here from the decommissioned portfolio repo on 2026-06-01.
- Deploy: `bash scripts/deploy_to_hostinger.sh` (production `/ai-minerals/`) or
  `bash scripts/deploy_to_hostinger.sh ai-minerals-beta` (staging).

## DO NOT BREAK THESE

1. **`index.qmd` must stay in `_quarto.yml`'s render list + navbar.** If it's
   missing, Quarto auto-generates a `<meta refresh>` redirect-to-`regional.html`
   stub as the index — that's the "root page just redirects" bug. The fix is
   always: index.qmd in the render list.
2. **The leak guard is mandatory.** Quarto's website render copies the WHOLE
   project tree into `_site/` — including `data/raw/` (the **family-private**
   drill-log scans), `src/`, `tools/`, `scripts/`. `deploy_to_hostinger.sh` and
   `deploy_internal.sh` use `rsync --exclude` (data/raw, tools, src, scripts,
   research, design, *.qmd) + `--delete-excluded` to keep those off the public
   server. **After any deploy, verify:** `curl -s -o /dev/null -w '%{http_code}'
   https://johnsondevco.com/ai-minerals/data/raw/bear_cub/SOURCE.md` must be
   **404**. (Two private-data leaks happened from this exact wholesale-copy; both
   were cleaned up. Don't reintroduce it.)
3. **No cross-repo includes / no rebuilding the umbrella.** Bear Cub and goldbug
   are **cross-site links**, not inlined chapters. Pulling them back in re-creates
   the deploy collision (two repos `--delete`-syncing `/ai-minerals/`) and drags
   their private data into this build.
4. **Only this repo deploys to `/ai-minerals/`.** If another session/repo also
   deploys there with `--delete`, they overwrite each other (last-writer-wins).

## Cross-site card targets (must stay 200)

- Bear Cub card → `https://johnsondevco.com/bearcub/bear_cub.html`
- goldbug card → `https://johnsondevco.com/goldbug/` (the **live tool**). NOT
  `/goldbug/goldbug.html` — `/goldbug/` is served by the live tool
  (Cloud-Run/aliased), not a static directory, so a dropped `goldbug.html` 404s
  there. The goldbug card is titled "live tool", so pointing at `/goldbug/` is correct.
- Deep notebooks: `/ai-minerals/notebooks/<region>/...` (ours),
  `/bearcub/notebooks/bear_cub/...` (Bear Cub's).

## Caching

HTML under `/ai-minerals/` is served `no-cache, must-revalidate, max-age=0` via a
`.htaccess` that **`deploy_to_hostinger.sh` (re)writes after each rsync** — the
`--delete` sync would otherwise wipe it, so don't remove that block. Effect: a
deploy is visible immediately, no hard-reload needed. Static assets (site_libs,
images, data) keep Hostinger's long default cache (they're stable).

Caveat: a page a browser cached BEFORE this change (e.g. the old redirect-index,
which Hostinger had served with `max-age=604800`) stays in that browser until the
copy expires or is cleared — the server cannot un-cache a copy a client already
holds. One-time fixes for a stuck page: DevTools → Network → "Disable cache" then
reload, or clear site data, or append `?v=1` to the URL.

## Open TODOs (safe to do later; site works without them)

- **301s** for the old umbrella URLs (anyone who has them): in the `/ai-minerals/`
  `.htaccess` — `bear_cub.html` → `/bearcub/bear_cub.html`,
  `goldbug.html` → `/goldbug/goldbug.html`.
- The deep-notebook site-map links in `index.qmd` point at `/ai-minerals/notebooks/`.
  Spot-check they 200 after a full render (exact subdirs/filenames).

## Resolved (2026-06-01)

- **goldbug writeup is live** at `/goldbug/goldbug.html` (with its iframe to the
  live tool). It's served from the NESTED `public_html/public_html/goldbug/` that
  the root `.htaccess` `^goldbug/(.*)` rewrite targets — NOT `~/goldbug/`
  (home-level, unserved). `gldbg/scripts/deploy_chapter_to_hostinger.sh MODE=final`
  deploys it there additively (beside the map). goldbug card + navbar link to it.
- Internal-site / `/plans/` 301s are written by `deploy_to_hostinger.sh` on
  production deploys.

## Pushing changes — the deploy routine (front door, chapters, OR notebooks)

The go-forward "notebook push" workflow. The site rebuilds + deploys as one unit;
you don't push a single notebook in isolation.

1. Edit the source: `index.qmd`, a chapter qmd, or a deep notebook under
   `notebooks/<region>/`.
2. **Render.** Full site: `uv run quarto render` (the deep notebooks need the venv;
   the `_freeze/` cache means only notebooks whose code changed re-execute, so it's
   usually fast). For an index/prose-only change, `quarto render index.qmd` is enough.
3. **Stage + look:** `bash scripts/deploy_to_hostinger.sh ai-minerals-beta`, then
   open `https://johnsondevco.com/ai-minerals-beta/`.
4. **Leak check — never skip:**
   `curl -s -o /dev/null -w '%{http_code}\n' https://johnsondevco.com/ai-minerals-beta/data/raw/bear_cub/SOURCE.md`
   must be **404** (also `/src/...`, `/tools/...`). If any is 200, stop and fix the
   rsync excludes before touching production.
5. **Production:** `bash scripts/deploy_to_hostinger.sh`. HTML is served no-cache,
   so changes appear immediately — no hard-reload needed.
6. Deep notebooks land at `/ai-minerals/notebooks/<region>/<page>.html` — reachable
   by direct URL, deliberately NOT surfaced in the navbar.

**Commits:** the bearcub + ai-minerals work shares ONE working tree on this
machine, so just `git commit` + `git push` normally — no rebase needed (that only
matters if you ever clone ai-minerals into a separate directory).

The `/ai-minerals-old/` directory on the server is a rollback snapshot of the
pre-cutover site — leave it until everything's settled.
