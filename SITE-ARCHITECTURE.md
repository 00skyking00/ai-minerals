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

## Caching gotcha

Hostinger serves these static pages with `cache-control: public, max-age=604800`
(7 days). After a deploy, returning visitors keep the OLD page for up to a week
unless they hard-reload. If you iterate often, set a shorter `max-age` for HTML
(server `.htaccess`) or accept the hard-reload.

## Open TODOs (safe to do later; site works without them)

- **301s** for the old umbrella URLs (anyone who has them): in the `/ai-minerals/`
  `.htaccess` — `bear_cub.html` → `/bearcub/bear_cub.html`,
  `goldbug.html` → `/goldbug/`.
- **goldbug writeup**: the prose chapter (`goldbug.qmd` in gldbg) has no clean
  public home — it only serves at `/goldbug-beta/goldbug.html`. `/goldbug/` is the
  live tool. gldbg decides whether to give the writeup a permanent URL. A stray
  `~/goldbug/goldbug.html` (a failed static-deploy attempt) sits unserved beside
  the live map and can be removed.
- The deep-notebook site-map links in `index.qmd` were repointed from the removed
  `/ai-minerals-internal/` to `/ai-minerals/notebooks/`. Spot-check they 200 after
  a full render (exact subdirs/filenames).

## How to tweak the front door safely

1. Edit `index.qmd` (cards, narrative, locator map).
2. `bash scripts/deploy_to_hostinger.sh ai-minerals-beta` → check
   `https://johnsondevco.com/ai-minerals-beta/` first.
3. Run the leak-guard check (item 2 above) on the beta.
4. Deploy production; hard-reload to see it past the 7-day cache.

The `/ai-minerals-old/` directory on the server is a rollback snapshot of the
pre-cutover site — leave it until you're sure everything's stable.
