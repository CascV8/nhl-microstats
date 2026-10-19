"""Turn cached play-by-play + shifts into per-player season microstats.

Shot types (the NHL logs no zone entries, so both are inferred from the
sequence of recorded events in the period)
- rush:  the puck was recorded outside the shooting team's attacking zone in
         the previous 8 s, with no faceoff since. ~6% of attempts -- play-by-play
         only catches rushes where something was logged in the neutral or
         defensive zone just before. 8 s rather than a tighter window because
         the lag between events and their logged times shifts between seasons:
         the high-scoring post-exit shots sit at 0-6 s in 2021-22 but 4-12 s
         in 2024-25.
- cycle: sustained offensive-zone play -- 10+ s since the puck was last
         recorded outside the zone or since a faceoff, and not a rebound or
         rush. Effectively "in-zone" offence; PBP can't separate a true cycle
         from a forecheck recovery.
Both use only event timing, so blocked attempts are classified too.

On-ice for/against comes from matching each event's period and clock to the
shift charts. Faceoffs count players starting a shift; other events count
players whose shift covers the second the event happened.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from puckscrape._util import start_year
from xg_model.collect import KEEP
from xg_model.features import BLUE_LINE_X, build_shots
from xg_model.predict import load_model

from . import collect, db

log = logging.getLogger("nhl_microstats.build")

ATTEMPTS = ("shot-on-goal", "missed-shot", "blocked-shot", "goal")
RUSH_SECONDS = 8
REBOUND_SECONDS = 3
CYCLE_SECONDS = 10
FIVE_ON_FIVE = "1551"


# --------------------------------------------------------------------------
# Shots
# --------------------------------------------------------------------------

def _zone_clocks(pbp: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """For each shot: seconds since the puck was last recorded outside the
    shooting team's attacking zone, since the last faceoff, and since the last
    shot attempt by either team."""
    df = pbp.sort_values(["game_id", "sort_order"])
    home_attacks_right = df["home_team_defending_side"] == "left"
    hx = df["x_coord"] * np.where(home_attacks_right, 1.0, -1.0)   # + = toward the home team's target net
    t = df["period_seconds"].astype("float")
    period = [df["game_id"], df["period"]]
    last = pd.DataFrame({
        "home": t.where((hx <= BLUE_LINE_X).fillna(False)),
        "away": t.where((hx >= -BLUE_LINE_X).fillna(False)),
        "faceoff": t.where(df["event_type"].isin(["faceoff", "period-start"])),
        "attempt": t.where(df["event_type"].isin(ATTEMPTS)),
    })
    # Only events *before* the shot count, so a shot from outside the blue line isn't its own rush.
    last = last.groupby(period).ffill().groupby(period).shift().join(df[["game_id", "event_id"]])
    m = shots[["game_id", "event_id", "is_home", "period_seconds"]].merge(last, on=["game_id", "event_id"], how="left")
    now = m["period_seconds"].to_numpy()
    out = np.where(m["is_home"] == 1, m["home"], m["away"])
    since = lambda last: now - np.nan_to_num(np.asarray(last, dtype=float), nan=-np.inf)
    return pd.DataFrame({"since_out": since(out), "since_faceoff": since(m["faceoff"]),
                         "since_attempt": since(m["attempt"])}, index=shots.index)


def score_shots(pbp: pd.DataFrame) -> pd.DataFrame:
    """Unblocked shots with xG from xg-model."""
    bundle = load_model()
    shots = build_shots(pbp[list(KEEP)])
    shots["xg"] = bundle["model"].predict_proba(shots[bundle["features"]])[:, 1]
    return shots[["game_id", "event_id", "xg"]]


def classify(pbp: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Rush and cycle flags for shot attempts (see the module docstring)."""
    c = _zone_clocks(pbp, ev)
    rush = (c["since_out"] <= RUSH_SECONDS) & (c["since_out"] < c["since_faceoff"])
    rebound = c["since_attempt"] <= REBOUND_SECONDS
    in_zone = np.minimum(c["since_out"], c["since_faceoff"]) >= CYCLE_SECONDS
    return pd.DataFrame({"rush": rush.astype(int), "cycle": (in_zone & ~rush & ~rebound).astype(int)})


def attempts(pbp: pd.DataFrame, shots: pd.DataFrame, team_of: pd.Series) -> pd.DataFrame:
    """Every shot attempt (blocked ones too), with the shooting team and shot flags."""
    ev = pbp[pbp["event_type"].isin(ATTEMPTS) & (pbp["period_type"] != "SO")].copy()
    ev["shooter_id"] = ev["shooting_player_id"].fillna(ev["scoring_player_id"])
    # Blocked shots are owned by the blocking team, so take the shooter's team from the shift charts.
    key = pd.MultiIndex.from_arrays([ev["game_id"], ev["shooter_id"].fillna(-1).astype("int64")])
    ev["team"] = team_of.reindex(key).to_numpy()
    ev["team"] = ev["team"].fillna(ev["event_team"].where(ev["event_type"] != "blocked-shot"))
    ev = ev.dropna(subset=["team"]).merge(shots, on=["game_id", "event_id"], how="left")
    ev["xg"] = ev["xg"].fillna(0)
    ev["is_home"] = (ev["team"] == ev["home_team"]).astype(int)
    ev[["rush", "cycle"]] = classify(pbp, ev)
    ev["is_goal"] = (ev["event_type"] == "goal").astype(int)
    ev["unblocked"] = (ev["event_type"] != "blocked-shot").astype(int)
    ev["sog"] = ev["event_type"].isin(["shot-on-goal", "goal"]).astype(int)
    ev["is_5v5"] = (ev["situation_code"].astype(str).str.zfill(4) == FIVE_ON_FIVE)
    return ev


# --------------------------------------------------------------------------
# Time on ice and on-ice events
# --------------------------------------------------------------------------

def toi(shifts: pd.DataFrame, skaters: set[int]) -> pd.DataFrame:
    """All-situations and 5v5 TOI (seconds) and games played per skater."""
    rows = []
    shifts = shifts[shifts["end_seconds"] > shifts["start_seconds"]]
    for (gid, period), s in shifts.groupby(["game_id", "period"], sort=False):
        n = int(s["end_seconds"].max()) + 1
        is_skater = s["player_id"].isin(skaters).to_numpy()
        teams = s["team_abbrev"].to_numpy()
        start, end = s["start_seconds"].to_numpy(), s["end_seconds"].to_numpy()
        five = np.ones(n - 1, dtype=bool)
        for team in np.unique(teams):
            for want, mask in ((5, is_skater), (1, ~is_skater)):
                d = np.zeros(n, dtype=int)
                sel = (teams == team) & mask
                np.add.at(d, start[sel], 1)
                np.add.at(d, end[sel], -1)
                five &= np.cumsum(d)[:-1] == want
        cum = np.concatenate([[0], np.cumsum(five)])
        rows.append(pd.DataFrame({"game_id": gid, "player_id": s["player_id"].to_numpy(),
                                  "toi_all": end - start, "toi_5v5": cum[end] - cum[start]}))
    t = pd.concat(rows, ignore_index=True)
    t = t[t["player_id"].isin(skaters)]
    return t.groupby("player_id").agg(gp=("game_id", "nunique"), toi_all=("toi_all", "sum"), toi_5v5=("toi_5v5", "sum"))


def on_ice(events: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, skater on the ice), with ``side`` = for/against."""
    out = []
    sh = shifts.groupby("game_id")
    for gid, ev in events.groupby("game_id", sort=False):
        if gid not in sh.groups:
            continue
        s = sh.get_group(gid)
        m = ev.reset_index(names="ev")[["ev", "period", "period_seconds"]].merge(s[["period", "player_id", "team_abbrev", "start_seconds", "end_seconds"]], on="period")
        t = m["period_seconds"]
        on = (m["start_seconds"] < t) & (t <= m["end_seconds"])
        out.append(m.loc[on, ["ev", "player_id", "team_abbrev"]])
    pairs = pd.concat(out, ignore_index=True)
    pairs["side"] = np.where(pairs["team_abbrev"].to_numpy() == events.loc[pairs["ev"], "team"].to_numpy(), "for", "against")
    return pairs.drop_duplicates(["ev", "player_id"])


# --------------------------------------------------------------------------
# Season build
# --------------------------------------------------------------------------

def _individual(ev: pd.DataFrame, goals_xg: pd.DataFrame) -> pd.DataFrame:
    by = ev.groupby("shooter_id")
    ind = pd.DataFrame({
        "goals": by["is_goal"].sum(),
        "ixg": by["xg"].sum(),
        "shot_attempts": by.size(),
        "shots_on_goal": by["sog"].sum(),
        "rush_shots": by["rush"].sum(),
        "cycle_shots": by["cycle"].sum(),
    })
    ind.index = ind.index.astype("int64")
    g = goals_xg
    a1 = g.groupby("assist1_player_id").agg(primary_assists=("event_id", "size"), primary_assist_xg=("xg", "sum"))
    a2 = g.groupby("assist2_player_id").agg(secondary_assists=("event_id", "size"))
    for a in (a1, a2):
        a.index = a.index.astype("int64")
    return ind.join([a1, a2], how="outer").fillna(0)


def _zone_puck_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """Takeaways and giveaways in the player's offensive and defensive zones.
    zone_code is relative to the team that owns the event, i.e. the player's team."""
    ev = pbp[pbp["event_type"].isin(["takeaway", "giveaway"]) & pbp["zone_code"].isin(["O", "D"])
             & pbp["player_id"].notna()]
    out = (ev.groupby(["player_id", ev["zone_code"].str.lower() + "z_" + ev["event_type"] + "s"])
           .size().unstack(fill_value=0))
    out.index = out.index.astype("int64")
    return out.reindex(columns=["oz_takeaways", "dz_takeaways", "oz_giveaways", "dz_giveaways"], fill_value=0)


def _onice_stats(ev: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.join(ev[["xg", "is_goal", "rush", "cycle"]], on="ev")
    p["shots"] = 1
    agg = p.groupby(["player_id", "side"])[["shots", "xg", "is_goal", "rush", "cycle"]].sum().unstack("side", fill_value=0)
    names = {"shots": "shots", "xg": "xg", "is_goal": "goals", "rush": "rush_shots", "cycle": "cycle_shots"}
    agg.columns = [f"{names[stat]}_{side}" for stat, side in agg.columns]
    return agg.rename(columns={"xg_for": "xgf", "xg_against": "xga"})


def build_season(season) -> None:
    y = start_year(season)
    log.info("%d-%02d: loading", y, (y + 1) % 100)
    pbp, shifts = collect.load_games(y)
    # The NHL has no shift charts for a few games (57 at the end of 2024-25). Without them there's
    # no TOI or on-ice data, so those games are left out entirely to keep rates consistent.
    no_shifts = set(pbp["game_id"]) - set(shifts["game_id"])
    if no_shifts:
        log.warning("%d-%02d: %d games have no shift charts and are skipped", y, (y + 1) % 100, len(no_shifts))
    pbp = pbp[(pbp["game_type"] == 2) & ~pbp["game_id"].isin(no_shifts)]
    # Some shift charts carry another game's shifts too (2025020565 has VGK-SJS in it).
    teams = pbp.drop_duplicates("game_id").melt("game_id", ["home_team", "away_team"], value_name="team_abbrev")
    shifts = shifts.merge(teams[["game_id", "team_abbrev"]].astype({"team_abbrev": "string"}), on=["game_id", "team_abbrev"])
    bios = pd.read_parquet(collect.DATA / f"bios_{y}.parquet")
    skaters = set(bios["player_id"].astype(int))

    team_of = shifts.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])["team_abbrev"]
    shots = score_shots(pbp)
    ev = attempts(pbp, shots, team_of)
    log.info("%d-%02d: %d attempts, %d unblocked scored", y, (y + 1) % 100, len(ev), len(shots))

    goals = ev[ev["is_goal"] == 1]
    times = toi(shifts, skaters)
    pairs = on_ice(ev, shifts)
    pairs = pairs[pairs["player_id"].isin(skaters)]

    frames = []
    pbp_5v5 = pbp["situation_code"].astype(str).str.zfill(4) == FIVE_ON_FIVE
    for strength, mask in (("all", slice(None)), ("5v5", ev["is_5v5"])):
        e = ev[mask] if strength == "5v5" else ev
        g = goals[goals["is_5v5"]] if strength == "5v5" else goals
        p = pairs[pairs["ev"].isin(e.index)]
        puck = _zone_puck_events(pbp[pbp_5v5] if strength == "5v5" else pbp)
        stats = _individual(e, g).join(_onice_stats(e, p), how="outer").join(puck, how="outer").fillna(0)
        stats = stats[stats.index.isin(times.index)]
        # Goals per shot on goal; players without a shot on goal have no shooting %.
        stats["shooting_pct"] = 100 * stats["goals"] / stats["shots_on_goal"].where(stats["shots_on_goal"] > 0)
        long = stats.rename_axis("player_id").reset_index().melt("player_id", var_name="stat").dropna(subset=["value"])
        frames.append(long.assign(season=y, strength=strength))
    stats = pd.concat(frames, ignore_index=True)

    # Team: every team the player suited up for that season, most games first.
    teams = (shifts.drop_duplicates(["game_id", "player_id"]).groupby("player_id")["team_abbrev"]
             .agg(lambda s: "/".join(s.value_counts().index)))
    ps_ = times.join(teams.rename("team")).join(bios.set_index("player_id")[["name", "position"]], how="inner")
    ps_ = ps_.reset_index().assign(season=y)

    edge = _edge(y, times["gp"])
    db.write_season(y, ps_, stats, edge)
    log.info("%d-%02d: %d skaters written", y, (y + 1) % 100, len(ps_))


# --------------------------------------------------------------------------
# NHL Edge
# --------------------------------------------------------------------------

EDGE_COLS = {
    "skating_speed_speed_max_imperial": "max_skating_speed",
    "skating_speed_bursts_over20_value": "bursts_over_20mph",
    "total_distance_skated_imperial": "distance_skated",
    "distance_max_game_imperial": "max_game_distance",
    "zone_time_details_offensive_zone_pctg": "oz_time_pct",
    "zone_time_details_offensive_zone_ev_pctg": "oz_time_pct_ev",
    "zone_time_details_neutral_zone_pctg": "nz_time_pct",
    "zone_time_details_defensive_zone_pctg": "dz_time_pct",
}


# Edge season counts (per-60 stats); the rest are maxima, averages or shares.
EDGE_COUNTS = {"bursts_over_20mph", "distance_skated"}


def _edge(y: int, gp: pd.Series | None = None) -> pd.DataFrame:
    """Edge stats in long format. ``gp`` is games played per player in our data:
    Edge counts every game, but games without shift charts are left out of
    TOI, so season counts are scaled to the games we have TOI for."""
    path = collect.DATA / f"edge_{y}.parquet"
    if not path.exists():
        log.warning("%d: no Edge data; run collect without --no-edge", y)
        return pd.DataFrame(columns=["player_id", "stat", "value"])
    edge = pd.read_parquet(path)
    have = {c: n for c, n in EDGE_COLS.items() if c in edge}
    if missing := set(EDGE_COLS) - set(have):
        log.warning("%d: Edge columns not found: %s", y, sorted(missing))
    e = edge[["player_id", *have]].rename(columns=have).astype({"player_id": "int64"})
    pct = [c for c in e if c.endswith("_pct") or c.endswith("_pct_ev")]
    e[pct] = e[pct].astype("float64") * 100        # Edge gives fractions
    if gp is not None and "info_games_played" in edge:
        edge_gp = edge["info_games_played"].astype("float64").to_numpy()
        share = (e["player_id"].map(gp).to_numpy() / edge_gp).clip(max=1)
        for c in EDGE_COUNTS & set(e):
            e[c] = e[c].astype("float64") * share
    shots = collect.DATA / f"edge_shots_{y}.parquet"
    if shots.exists():
        shots = pd.read_parquet(shots)[["player_id", "avg_shot_speed"]].rename(columns={"avg_shot_speed": "shot_strength"})
        e = e.merge(shots, on="player_id", how="outer")
    else:
        log.warning("%d: no Edge shot speeds; run collect without --no-edge", y)
    return e.melt("player_id", var_name="stat").dropna(subset=["value"]).astype({"value": "float64"})


def build(seasons) -> None:
    db.init()
    for s in seasons:
        build_season(s)
    from .curves import fit_all
    fit_all()
