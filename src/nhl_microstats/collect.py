"""Download and cache what the microstats are built from.

Per game: play-by-play (the xg-model columns plus assists) and shift charts,
cached as parquet under data/games/<season>/. Per season: player bios (for
positions) and NHL Edge skater summaries. Everything resumes where it left off.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import puckscrape as ps
from puckscrape._util import start_year
from xg_model.collect import KEEP, game_ids

log = logging.getLogger("nhl_microstats.collect")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# player_id is the player credited with a takeaway or giveaway; blocking/hitting/hittee for blocks and hits;
# committed_by/drawn_by, the penalty type (MIN, MAJ, MIS, ...) and its description for penalties.
PBP_KEEP = KEEP | {"assist1_player_id": "Int64", "assist2_player_id": "Int64", "player_id": "Int64",
                   "blocking_player_id": "Int64", "hitting_player_id": "Int64", "hittee_player_id": "Int64",
                   "committed_by_player_id": "Int64", "drawn_by_player_id": "Int64",
                   "details_type_code": "string", "desc_key": "string"}
SHIFT_KEEP = {"player_id": "int64", "team_abbrev": "string", "period": "int64",
              "start_seconds": "int64", "end_seconds": "int64"}


def _slim(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, dtype in cols.items():
        out[col] = (df[col] if col in df else pd.Series(pd.NA, index=df.index)).astype(dtype)
    return out


def game_dir(season) -> Path:
    return DATA / "games" / str(start_year(season))


def _fetch(game_id: int, folder: Path) -> None:
    pbp = _slim(ps.nhl.play_by_play(game_id), PBP_KEEP)
    if (folder / f"{game_id}_pbp.parquet").exists():   # refreshing an older pbp file; shifts are fine
        pbp.to_parquet(folder / f"{game_id}_pbp.parquet")
        return
    shifts = ps.nhl.shifts(game_id)
    shifts = _slim(shifts, SHIFT_KEEP) if len(shifts) else pd.DataFrame(columns=list(SHIFT_KEEP))
    # Shifts first: a game counts as cached once its pbp file exists.
    shifts.to_parquet(folder / f"{game_id}_shifts.parquet")
    pbp.to_parquet(folder / f"{game_id}_pbp.parquet")


def _stale(path: Path) -> bool:
    """Not downloaded yet, or cached before a column was added to PBP_KEEP."""
    return not path.exists() or not set(PBP_KEEP) <= set(pq.read_schema(path).names)


def _retry_missing_shifts(folder: Path) -> list[int]:
    """The NHL sometimes posts shift charts days after a game; re-ask for any cached empty.
    Returns the games that have them now."""
    empty = [p for p in folder.glob("*_shifts.parquet") if pq.read_metadata(p).num_rows == 0]
    found = []
    for p in empty:
        try:
            shifts = ps.nhl.shifts(int(p.stem.split("_")[0]))
        except Exception as e:
            log.debug("shifts %s failed: %s", p.stem, e)
            continue
        if len(shifts):
            _slim(shifts, SHIFT_KEEP).to_parquet(p)
            found.append(int(p.stem.split("_")[0]))
    if empty:
        log.info("%s: %d of %d games without shift charts now have them", folder.name, len(found), len(empty))
    return found


def collect_games(season, workers: int = 8) -> list[int]:
    """Download every completed game not cached yet, and retry missing shift charts.
    Returns the games that changed: newly downloaded, or shift charts that arrived."""
    folder = game_dir(season)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [g for g in game_ids(season) if _stale(folder / f"{g}_pbp.parquet")]
    log.info("%s: %d games to download", season, len(todo))
    changed = _retry_missing_shifts(folder)
    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_fetch, g, folder): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
                changed.append(futures[fut])
            except Exception as e:  # one bad game shouldn't sink the season
                failed.append(futures[fut])
                log.warning("game %s failed: %s", futures[fut], e)
            if i % 200 == 0:
                log.info("%s: %d/%d", season, i, len(todo))
    if failed:
        log.warning("%d games failed; rerun to retry: %s", len(failed), failed[:10])
    return changed


def collect_bios(season, refresh: bool = False) -> Path:
    """Name and position for every skater who played that season. ``refresh`` re-downloads,
    for a season in progress (call-ups and debuts)."""
    path = DATA / f"bios_{start_year(season)}.parquet"
    if refresh or not path.exists():
        bios = ps.nhl.skater_stats(season, report="bios")
        bios = bios.rename(columns={"player_id": "player_id", "skater_full_name": "name", "position_code": "position"})
        bios[["player_id", "name", "position"]].to_parquet(path)
    return path


def collect_edge(season, player_ids, workers: int = 4, refresh=()) -> Path:
    """NHL Edge skater summaries (speed, shot speed, distance, zone time). Players in
    ``refresh`` are downloaded again (their season totals grow with every game)."""
    path = DATA / f"edge_{start_year(season)}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["player_id"])
    have = have[~have["player_id"].astype(int).isin(set(map(int, refresh)))]
    todo = sorted(set(int(p) for p in player_ids) - set(have["player_id"].astype(int)))
    log.info("%s: %d Edge summaries to download", season, len(todo))
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(ps.edge.skater_summary, p, season): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # players with no Edge data 404
                log.debug("edge %s failed: %s", futures[fut], e)
            if i % 200 == 0:
                log.info("%s edge: %d/%d", season, i, len(todo))
    if rows:
        new = pd.concat([have, *rows], ignore_index=True)
        new = new.astype({c: "string" for c in new.select_dtypes("object").columns}).astype({"player_id": "int64"})
        new.to_parquet(path)
    return path


def _shot_speed(player_id: int, season) -> dict:
    d = ps.edge.raw("skater-shot-speed-detail", player_id, season).get("shotSpeedDetails", {})
    return {"player_id": player_id, "avg_shot_speed": (d.get("avgShotSpeed") or {}).get("imperial")}


def collect_shot_speed(season, player_ids, workers: int = 4, refresh=()) -> Path:
    """Average shot speed from NHL Edge (the summary only has the hardest shot)."""
    path = DATA / f"edge_shots_{start_year(season)}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["player_id", "avg_shot_speed"])
    have = have[~have["player_id"].astype(int).isin(set(map(int, refresh)))]
    todo = sorted(set(int(p) for p in player_ids) - set(have["player_id"].astype(int)))
    log.info("%s: %d Edge shot speeds to download", season, len(todo))
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_shot_speed, p, season): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # players with no Edge data 404
                log.debug("edge shot speed %s failed: %s", futures[fut], e)
            if i % 200 == 0:
                log.info("%s edge shot speed: %d/%d", season, i, len(todo))
    if rows:
        new = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        new.astype({"player_id": "int64", "avg_shot_speed": "float64"}).to_parquet(path)
    return path


def load_games(season) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = game_dir(season)
    pbp_files = sorted(folder.glob("*_pbp.parquet"))
    if not pbp_files:
        raise FileNotFoundError(f"no cached games for {season}; run `nhl-microstats collect {season}` first")
    pbp = pd.concat((pd.read_parquet(f) for f in pbp_files), ignore_index=True)
    shifts = pd.concat((pd.read_parquet(f).assign(game_id=int(f.stem.split("_")[0]))
                        for f in sorted(folder.glob("*_shifts.parquet"))), ignore_index=True)
    shifts = shifts.dropna(subset=["player_id"]).astype(SHIFT_KEEP)   # games without shift charts
    # The shift charts endpoint sometimes lists the same shift more than once.
    shifts = (shifts.sort_values("end_seconds")
              .drop_duplicates(["game_id", "player_id", "period", "start_seconds"], keep="last")
              .reset_index(drop=True))
    return pbp, shifts
