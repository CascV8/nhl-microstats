"""Static website: an index of every skater with category percentiles, and a player page
per skater that reuses the card. No server; any static host can serve ``site/``.

    nhl-microstats site                 # current season + last complete season -> site/
    nhl-microstats site 2026 2025 2024 --out public

Layout of the output:

    index.html                  player directory and leaderboards
    player.html                 the card, loaded from ?season=YYYY&id=NNN
    data/meta.json              seasons, when it was built, games included, games awaiting shift charts
    data/<season>/index.json    one compact row per skater
    data/<season>/players/<id>.json   one card's data (the same shape the cards page embeds)

Everything is written to a temporary folder first and swapped in at the end, so a
visitor never sees half an update.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from puckscrape._util import start_year

from . import collect
from .cards import MODES, TEMPLATE, card_data
from .percentile import complete_seasons, season_name

log = logging.getLogger("nhl_microstats.site")

INDEX_TEMPLATE = Path(__file__).with_name("site_index.html")
OUT = collect.ROOT / "site"


def default_seasons() -> list[int]:
    """The season in progress (if any) and the last complete one."""
    from .db import connect
    with connect() as con:
        built = [r[0] for r in con.execute("SELECT DISTINCT season FROM player_seasons ORDER BY season")]
    done = complete_seasons()
    current = [y for y in built if y not in done and y > max(done)]
    return [*current[-1:], max(done)]


def games_status(y: int) -> dict:
    """Games cached for a season, and which of them still have no shift charts. Past seasons'
    games aren't kept where the site is built (CI), so their count is left out there."""
    folder = collect.game_dir(y)
    if not folder.exists():
        return {"games": None, "awaiting_shifts": []}
    shifts = sorted(folder.glob("*_shifts.parquet"))
    missing = [int(p.stem.split("_")[0]) for p in shifts if pq.read_metadata(p).num_rows == 0]
    return {"games": len(list(folder.glob("*_pbp.parquet"))), "awaiting_shifts": missing}


def _index_row(c: dict) -> dict:
    """The few numbers the directory shows, per mode."""
    modes = {}
    for mode in MODES:
        h = c["modes"][mode].get("xgf_pct")
        modes[mode] = {"xgf": [h["v"], h["pct"]] if h else None,
                       "cats": {title: pct for title, _, _, pct in c["categories"][mode]}}
    return {"id": c["id"], "name": c["name"], "team": c["team"], "pos": c["position"], "group": c["group"],
            "gp": c["gp"], "toi": c["toi"], "g": c["g"], "a": c["a"], "modes": modes}


def export_site(seasons=None, out: Path | str = OUT) -> Path:
    out = Path(out)
    seasons = [start_year(s) for s in seasons] if seasons else default_seasons()
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data").mkdir(parents=True)

    meta = {"built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "seasons": []}
    for y in seasons:
        d = card_data(None, y)
        cards = d.pop("cards")
        folder = tmp / "data" / str(y)
        (folder / "players").mkdir(parents=True)
        for c in cards:
            (folder / "players" / f"{c['id']}.json").write_text(json.dumps(d | {"site": True, "cards": [c]}), encoding="utf-8")
        index = d | {"rows": [_index_row(c) for c in cards]}
        (folder / "index.json").write_text(json.dumps(index), encoding="utf-8")
        meta["seasons"].append({"season": y, "name": season_name([y]), "early": d["early"], "players": len(cards),
                                **games_status(y)})
        log.info("%s: %d players", season_name([y]), len(cards))
    (tmp / "data" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.copy(INDEX_TEMPLATE, tmp / "index.html")
    # The card template is a fragment (cards.html gets wrapped by whatever shows it); give it a document head.
    head = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    (tmp / "player.html").write_text(head + TEMPLATE.read_text(encoding="utf-8").replace(
        "<title>NHL Microstat Cards</title>", "<title>Player card · NHL Microstats</title>", 1), encoding="utf-8")

    old = out.with_name(out.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    if old.exists():
        shutil.rmtree(old)
    log.info("site written to %s", out.resolve())
    return out
