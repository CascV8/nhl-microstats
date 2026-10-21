"""Static website: an index of every skater with category percentiles, and a player page
per skater that reuses the card. No server; any static host can serve ``site/``.

    nhl-microstats site                 # 2023-24 on (the curves' pool) -> site/
    nhl-microstats site 2026 2025 --out public

Layout of the output:

    index.html                  player directory and leaderboards
    player.html                 the card, loaded from ?season=YYYY&id=NNN
    data/meta.json              seasons, when it was built, games included, games awaiting shift charts
    data/player_seasons.json    {player id: [seasons with a card]}, for the player page's season picker
    data/<season>/index.json    one compact row per skater
    data/<season>/players/<id>.json   one card's data (the same shape the cards page embeds)

Everything is written to a temporary folder first and swapped in at the end, so a
visitor never sees half an update. Complete seasons don't change between games, so their
files are cached in data/site_cache/ and reused until the code or that season's data changes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from puckscrape._util import start_year

from . import collect
from .cards import MODES, TEMPLATE, card_data
from .db import connect
from .percentile import complete_seasons, season_name

log = logging.getLogger("nhl_microstats.site")

INDEX_TEMPLATE = Path(__file__).with_name("site_index.html")
OUT = collect.ROOT / "site"
CACHE = collect.DATA / "site_cache"
# What a season's exported files depend on, besides its own rows: the code and the curves.
CODE = ["cards.py", "curves.py", "percentile.py", "site.py", "db.py"]


FIRST_SEASON = 2023   # the site starts with the first season of the curves' comparison pool (2023-24)


def default_seasons() -> list[int]:
    """Every season in the database from FIRST_SEASON on, newest first."""
    with connect() as con:
        return [r[0] for r in con.execute("SELECT DISTINCT season FROM player_seasons WHERE season >= ? "
                                          "ORDER BY season DESC", [FIRST_SEASON])]


def _stamp(y: int) -> str:
    """Changes whenever season ``y``'s exported files could: its data, the curves, or the code."""
    h = hashlib.sha256()
    for name in CODE:
        h.update(Path(__file__).with_name(name).read_bytes())
    with connect() as con:
        for q, params in (("SELECT count(*), total(value) FROM stats WHERE season = ?", [y]),
                          ("SELECT count(*), total(toi_all), group_concat(team) FROM player_seasons WHERE season = ?", [y]),
                          ("SELECT count(*), total(a), total(b), total(c) FROM curves", [])):
            h.update(repr(con.execute(q, params).fetchone()).encode())
    return h.hexdigest()


def _export_season(y: int, folder: Path) -> dict:
    """Write one season's index and player files into ``folder``; returns its meta entry."""
    d = card_data(None, y)
    cards = d.pop("cards")
    (folder / "players").mkdir(parents=True)
    for c in cards:
        (folder / "players" / f"{c['id']}.json").write_text(json.dumps(d | {"site": True, "cards": [c]}), encoding="utf-8")
    (folder / "index.json").write_text(json.dumps(d | {"rows": [_index_row(c) for c in cards]}), encoding="utf-8")
    return {"season": y, "name": season_name([y]), "early": d["early"], "players": len(cards)}


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
        o = (c.get("overall") or {}).get(mode)
        modes[mode] = {"overall": [o[2], o[0]] if o else None,
                       "cats": {title: [pct, avg] for title, avg, _, pct in c["categories"][mode]}}
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
    done = set(complete_seasons())
    for y in sorted(seasons, reverse=True):
        folder = tmp / "data" / str(y)
        cached = CACHE / str(y)
        if y in done and (cached / "stamp").exists() and (cached / "stamp").read_text() == _stamp(y):
            shutil.copytree(cached / "files", folder)
            entry = json.loads((cached / "meta.json").read_text())
            log.info("%s: reused (unchanged)", entry["name"])
        else:
            entry = _export_season(y, folder)
            log.info("%s: %d players", entry["name"], entry["players"])
            if y in done:     # keep it for next time
                if cached.exists():
                    shutil.rmtree(cached)
                shutil.copytree(folder, cached / "files")
                (cached / "meta.json").write_text(json.dumps(entry))
                (cached / "stamp").write_text(_stamp(y))
        meta["seasons"].append(entry | games_status(y))
    (tmp / "data" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with connect() as con:
        rows = con.execute(f"SELECT player_id, season FROM player_seasons WHERE season IN ({','.join('?' * len(seasons))}) "
                           "ORDER BY season DESC", seasons).fetchall()
    by_player = {}
    for pid, y in rows:
        by_player.setdefault(str(pid), []).append(y)
    (tmp / "data" / "player_seasons.json").write_text(json.dumps(by_player), encoding="utf-8")
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
