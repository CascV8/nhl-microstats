# Website design

Goal: a public site with every skater's microstat card, updated as soon as the NHL
releases each game's sheet.

## Shape: a static site plus a watcher

```
NHL API ──> GitHub Actions (nhl-microstats update) ──> microstats.db + current season ──> site/ ──> Cloudflare Pages
            every 5 min                                 (Actions cache, ~120 MB)             (15 MB)
```

There is no application server and no database to host. Every page is a static file,
and the watcher regenerates them whenever there's something new. Reasons:

- **Cheap and fast to serve.** Any static host or CDN works, and nothing can be queried
  into slowness or downtime.
- **The expensive part is offline anyway.** Building a season takes seconds to minutes
  of pandas over 850 MB of cached play-by-play. Doing it per request would need a
  real server; doing it once per game needs a cron job.
- **Visitors never see half an update.** The export writes to `site.tmp/` and swaps it in.

## Pages

| Page | What it shows |
|---|---|
| `index.html` | Every skater in a season: search, F/D filter, team, minimum GP, 5v5 / all situations, and a sortable table of xGF% and each category's percentile. Season picker; status line with games included, games awaiting shift charts, and last update. |
| `player.html?season=2026&id=8476902` | The player card (`cards_template.html`, the same file `nhl-microstats cards` uses) loaded with that player's JSON. Back link to the index. |

The card template works both ways: `nhl-microstats cards` embeds the data; on the site,
`DATA` is empty and the page fetches `data/<season>/players/<id>.json`.

## Data files

```
data/meta.json                      built time; per season: name, players, games, games awaiting shift charts
data/<season>/index.json            one compact row per skater (what the index table needs)
data/<season>/players/<id>.json     one card, the same shape the cards page embeds (~3 KB)
```

Seasons from 2023-24 (the first season of the comparison pool) on are exported by default (about 3,450
player files, 32 MB). Adding seasons is `nhl-microstats site 2026 2025 2024 ...`.
`card_data` builds every player from `percentile.season_table`, which ranks a whole
season in one pass per stat (about 40 s per season instead of hours of per-player queries).
A test checks that it matches `player_table` exactly.

## When does it update?

The NHL's game state goes `LIVE` → `FINAL` when a game ends → `OFF` once the game sheet
is checked, usually within the hour. **`OFF` is the trigger**: play-by-play is final then.

Each `watch` pass (default every 5 minutes):

1. Asks the schedule for the last two days' games (one request). Nothing newly `OFF`, and
   no shift-chart retry due: **done**. That's the normal idle cost.
2. A newly official game: download its play-by-play and shift charts, and retry any games
   still missing shift charts. Retries alone happen at most every 30 minutes.
3. If anything changed: refresh bios (call-ups), re-fetch NHL Edge for everyone who played
   in the changed games (Edge season totals grow every game), rebuild the current season,
   re-export the site, then run the `--deploy` command.

**No curve refit.** The season in progress is never part of a comparison pool, so its
games can't move the curves. Refit (`nhl-microstats fit`) once a season completes.

Typical latency from game sheet to live site: up to 5 minutes polling + about 2 minutes of
work + deploy time. Shift charts the NHL hasn't posted hold a game out of the stats until
they arrive (29 games from Sep 30 to Oct 3, 2026 are still missing).

## Hosting: GitHub Actions + Cloudflare Pages

Both free. `.github/workflows/update-site.yml` runs every 5 minutes on GitHub's runners
(unlimited minutes for a public repo) and deploys `site/` to Cloudflare Pages with wrangler.

- **What it keeps between runs.** Only what an update needs: `microstats.db` (past seasons'
  stats and the curves), the current season's games, bios and Edge, and the xG model, in the
  Actions cache (~120 MB by season's end). If the cache is empty (first run, or GitHub evicted
  it after 7 days unused) it downloads `microstats.db` and `xg.joblib` from the `data-bootstrap`
  release and re-downloads the current season, which takes a few minutes.
- **Deploys only when something changed**, and saves the cache only after a successful
  deploy, so a failed deploy is retried on the next run.
- **Who starts it.** GitHub's own schedule is best-effort: set to every 5 minutes, it ran
  every 3-5 hours. So a Cloudflare Worker with a cron trigger (`scheduler/`, free) calls
  GitHub's API to start the workflow every 5 minutes; the workflow's own hourly schedule
  is a fallback. The Worker needs a fine-grained GitHub token (this repo, *Actions: Read and
  write*) as its secret `GITHUB_TOKEN`; tokens expire, at most a year out, so renew it then.
  Deploy changes to it with `npx wrangler deploy` in `scheduler/`.
- **Latency.** A game reaches the site within about 5-10 minutes of the NHL marking it
  official (the next 5-minute run, plus a couple of minutes to rebuild and deploy).
- **Keepalive.** GitHub turns off schedules in repos with no activity for 60 days; the
  workflow pushes an empty commit when the last one is 45 days old.
- **Run by hand:** Actions tab -> Update site -> Run workflow (tick "force" to rebuild anyway).

One-time setup: a Cloudflare API token with *Cloudflare Pages: Edit*, stored as the repo
secrets `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. The first deploy creates the
Pages project (`nhl-microstats`, or the repo variable `PAGES_PROJECT`), served at
`https://<project>.pages.dev`.

**After a season ends:** run `build` and `fit` locally (that's when curves change), then
replace `microstats.db` on the `data-bootstrap` release and clear the Actions cache.

Not covered yet: the goal-tracking stats (`gt_*`) aren't refreshed by `update`, so the
current season's are as of the last local `goal-tracking` run.

## Open questions

- Which seasons to publish (default: current + last complete).
- Branding and the site name.
- Whether to add team pages, leaderboards per stat, or player-vs-player comparisons next.
