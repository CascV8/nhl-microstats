"""Keep the website current: as soon as the NHL marks a game official, pull it in and
re-export the site.

    nhl-microstats update                       # one pass: new games -> rebuild -> site/
    nhl-microstats watch --every 5              # poll every 5 minutes, forever
    nhl-microstats watch --deploy "npx wrangler pages deploy site --project-name microstats"

A game goes LIVE -> FINAL when it ends and -> OFF once the NHL has checked the game sheet
(usually within the hour). The play-by-play is complete at OFF; shift charts usually
arrive with it but sometimes days later, so every pass also re-asks for the ones missing.
A game without shift charts stays out of the stats until they arrive.

A pass with nothing new costs one schedule request. With something new it downloads the
games, re-fetches NHL Edge for everyone who played in them (Edge season totals grow every
game), refreshes bios (call-ups), rebuilds the current season and re-exports the site.
Percentile curves are not refit: the season in progress is never part of a comparison
pool, so its games can't move the curves.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import time

import pandas as pd
import puckscrape as ps
import pyarrow.parquet as pq

from . import collect
from .build import build_season
from .site import OUT, default_seasons, export_site

log = logging.getLogger("nhl_microstats.update")

OFFICIAL = {"OFF"}             # the game sheet is final
SHIFT_RETRY_MINUTES = 30       # how often to re-ask for shift charts the NHL hasn't posted
STATE = collect.DATA / "update_state.json"   # kept with the data, so it survives between CI runs


def _last_retry() -> float:
    try:
        return float(json.loads(STATE.read_text())["last_shift_retry"])
    except (OSError, ValueError, KeyError):
        return 0.0


def _retried_now() -> None:
    STATE.write_text(json.dumps({"last_shift_retry": time.time()}))


def current_season(today: dt.date | None = None) -> int:
    """Start year of the season under way (seasons start in the fall)."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def _recent_official(y: int, days: int = 2) -> list[int]:
    """Regular-season games of season ``y`` from the last few days that the NHL marks official."""
    today = dt.date.today()
    s = ps.nhl.schedule(today - dt.timedelta(days=days), today)
    if s.empty:
        return []
    s = s[(s["season"] == int(f"{y}{y + 1}")) & (s["game_type"] == 2) & s["game_state"].isin(OFFICIAL)]
    return sorted(s["game_id"].astype(int))


def _players_in(y: int, games: list[int]) -> set[int]:
    folder = collect.game_dir(y)
    ids = set()
    for g in games:
        path = folder / f"{g}_shifts.parquet"
        if path.exists():
            ids |= set(pd.read_parquet(path, columns=["player_id"])["player_id"].dropna().astype(int))
    return ids


def update_once(season=None, out=OUT, deploy: str | None = None, force: bool = False) -> bool:
    """One pass. Returns True if the site was rebuilt."""
    y = int(season) if season else current_season()
    folder = collect.game_dir(y)
    pending = [g for g in _recent_official(y) if not (folder / f"{g}_pbp.parquet").exists()]
    retry = time.time() - _last_retry() >= SHIFT_RETRY_MINUTES * 60
    awaiting = retry and any(pq.read_metadata(p).num_rows == 0 for p in folder.glob("*_shifts.parquet"))
    if pending or force:              # full pass: new games, and missing shift charts retried along the way
        changed = collect.collect_games(y)
        _retried_now()
    elif awaiting:                    # only re-ask for shift charts
        changed = collect._retry_missing_shifts(folder)
        _retried_now()
    else:
        log.info("nothing new")
        return False
    if not changed and not force:
        log.info("no new games or shift charts yet")
        return False
    log.info("%d games changed: %s", len(changed), changed[:10])

    bios = pd.read_parquet(collect.collect_bios(y, refresh=True))
    played = _players_in(y, changed)
    collect.collect_edge(y, bios["player_id"], refresh=played)
    collect.collect_shot_speed(y, bios["player_id"], refresh=played)
    build_season(y)
    export_site(sorted({y, *default_seasons()}, reverse=True), out)   # past seasons come from the cache
    if deploy:
        log.info("deploying: %s", deploy)
        subprocess.run(deploy, shell=True, check=True)
    return True


def watch(every_minutes: float = 5, **kwargs) -> None:
    """Run ``update_once`` forever. Errors are logged and retried next pass."""
    log.info("watching for official games every %g min", every_minutes)
    while True:
        try:
            update_once(**kwargs)
        except Exception:     # a network blip shouldn't stop the watcher
            log.exception("update failed; retrying next pass")
        time.sleep(every_minutes * 60)
