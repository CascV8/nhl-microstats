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

High danger: an unblocked attempt from the inner slot (within 25 ft of the net
and 10 ft of the middle), a rebound, or a quick shot off a rush (both as the
xG model defines them).

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
SLOT_DISTANCE, SLOT_WIDTH = 25, 10   # feet


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
    """Unblocked shots with xG from xg-model, and whether each was high danger."""
    bundle = load_model()
    shots = build_shots(pbp[list(KEEP)])
    shots["xg"] = bundle["model"].predict_proba(shots[bundle["features"]])[:, 1]
    slot = (shots["distance"] <= SLOT_DISTANCE) & (shots["y_abs"] <= SLOT_WIDTH)
    shots["hd"] = (slot | (shots["rebound"] == 1) | (shots["rush"] == 1)).astype(int)
    return shots[["game_id", "event_id", "xg", "hd"]]


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
    ev["hd_xg"] = ev["xg"] * ev["hd"].fillna(0)
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

def toi(shifts: pd.DataFrame, skaters: set[int], by_game: pd.DataFrame | None = None) -> pd.DataFrame:
    """All-situations and 5v5 TOI (seconds) and games played per skater."""
    t = toi_by_game(shifts, skaters) if by_game is None else by_game
    return t.groupby("player_id").agg(gp=("game_id", "nunique"), toi_all=("toi_all", "sum"), toi_5v5=("toi_5v5", "sum"))


def _five_on_five(s: pd.DataFrame, skaters: set[int]) -> np.ndarray:
    """For one game-period's shifts: which seconds have five skaters and a goalie a side."""
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
    return five


def toi_by_game(shifts: pd.DataFrame, skaters: set[int]) -> pd.DataFrame:
    """Per-shift TOI rows: game, player, team, all-situations and 5v5 seconds."""
    rows = []
    shifts = shifts[shifts["end_seconds"] > shifts["start_seconds"]]
    for (gid, period), s in shifts.groupby(["game_id", "period"], sort=False):
        teams = s["team_abbrev"].to_numpy()
        start, end = s["start_seconds"].to_numpy(), s["end_seconds"].to_numpy()
        cum = np.concatenate([[0], np.cumsum(_five_on_five(s, skaters))])
        rows.append(pd.DataFrame({"game_id": gid, "player_id": s["player_id"].to_numpy(), "team": teams,
                                  "toi_all": end - start, "toi_5v5": cum[end] - cum[start]}))
    t = pd.concat(rows, ignore_index=True)
    return t[t["player_id"].isin(skaters)]


def competition(shifts: pd.DataFrame, skaters: set[int], xgf: pd.Series, xga: pd.Series) -> pd.Series:
    """Quality of competition: the average 5v5 xGF% of the opposing skaters a player was
    on the ice against, each at his actual season xGF%, weighted by the 5v5 seconds they
    shared. ``xgf``/``xga`` are each skater's on-ice 5v5 season totals. An opponent with
    no 5v5 xG either way has no xGF% and is left out."""
    ids = xgf.index.union(xga.index)
    xgf, xga = xgf.reindex(ids).fillna(0), xga.reindex(ids).fillna(0)
    share = (100 * xgf / (xgf + xga)).where(xgf + xga > 0)
    num, den = [], []
    shifts = shifts[shifts["end_seconds"] > shifts["start_seconds"]]
    for _, s in shifts.groupby(["game_id", "period"], sort=False):
        five = _five_on_five(s, skaters)
        sk = s[s["player_id"].isin(skaters)]
        ids, idx = np.unique(sk["player_id"].to_numpy(), return_inverse=True)
        d = np.zeros((len(ids), len(five) + 1))
        np.add.at(d, (idx, sk["start_seconds"].to_numpy()), 1)
        np.add.at(d, (idx, sk["end_seconds"].to_numpy()), -1)
        on = (np.cumsum(d, axis=1)[:, :-1] > 0) & five
        team = sk.drop_duplicates("player_id").set_index("player_id")["team_abbrev"].reindex(ids).to_numpy()
        opp_share = share.reindex(ids).to_numpy()
        rated = ~np.isnan(opp_share)
        for t in np.unique(team):
            me, them = team == t, (team != t) & rated
            shared = on[me].astype(float) @ on[them].T.astype(float)     # seconds together, me x them
            num.append(pd.Series(shared @ opp_share[them], index=ids[me]))
            den.append(pd.Series(shared.sum(axis=1), index=ids[me]))
    num, den = pd.concat(num).groupby(level=0).sum(), pd.concat(den).groupby(level=0).sum()
    return (num / den)[den > 0].rename("qoc_xgf_pct")


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
        "fenwick": by["unblocked"].sum(),
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
    """Takeaways and giveaways in all zones, and in the player's offensive and defensive zones.
    zone_code is relative to the team that owns the event, i.e. the player's team."""
    ev = pbp[pbp["event_type"].isin(["takeaway", "giveaway"]) & pbp["player_id"].notna()]
    total = ev.groupby(["player_id", ev["event_type"] + "s"]).size().unstack(fill_value=0)
    ev = ev[ev["zone_code"].isin(["O", "D"])]
    zones = (ev.groupby(["player_id", ev["zone_code"].str.lower() + "z_" + ev["event_type"] + "s"])
             .size().unstack(fill_value=0))
    out = total.join(zones, how="outer").fillna(0)
    out.index = out.index.astype("int64")
    return out.reindex(columns=["takeaways", "giveaways", "oz_takeaways", "dz_takeaways", "oz_giveaways", "dz_giveaways"],
                       fill_value=0)


GIVEAWAY_SECONDS = 10


def _giveaway_xga(pbp: pd.DataFrame, shots: pd.DataFrame) -> dict[str, pd.Series]:
    """xG of the opponent's unblocked shots in the 10 s after each of a player's giveaways
    (same period, no faceoff in between), by the strength the giveaway happened at."""
    p = pbp.merge(shots[["game_id", "event_id", "xg"]], on=["game_id", "event_id"], how="left")
    p = p.sort_values(["game_id", "sort_order"])
    pid_all, xga_all, is_five = [], [], []
    for _, d in p.groupby(["game_id", "period"], sort=False):
        typ = d["event_type"].to_numpy()
        gv = np.flatnonzero((typ == "giveaway") & d["player_id"].notna().to_numpy())
        if not len(gv):
            continue
        t, team = d["period_seconds"].to_numpy(), d["event_team"].fillna("").astype(object).to_numpy()
        xg, pid = d["xg"].to_numpy(dtype=float), d["player_id"].to_numpy()
        f = (d["situation_code"].astype(str).str.zfill(4) == FIVE_ON_FIVE).to_numpy()
        for n in gv:
            s = 0.0
            for j in range(n + 1, len(d)):
                if t[j] - t[n] > GIVEAWAY_SECONDS or typ[j] == "faceoff":
                    break
                if team[j] != team[n] and xg[j] == xg[j]:   # the opponent's unblocked shot (xg not NaN)
                    s += xg[j]
            pid_all.append(int(pid[n])); xga_all.append(s); is_five.append(bool(f[n]))
    g = pd.DataFrame({"player_id": pid_all, "xga": xga_all, "five": is_five})
    return {"all": g.groupby("player_id")["xga"].sum().rename("giveaway_xga"),
            "5v5": g[g["five"]].groupby("player_id")["xga"].sum().rename("giveaway_xga")}


def _blocks_hits(pbp: pd.DataFrame) -> pd.DataFrame:
    """Shots blocked, hits thrown and hits taken, per player."""
    hits = pbp[pbp["event_type"] == "hit"]
    out = pd.concat({"blocks": pbp.loc[pbp["event_type"] == "blocked-shot", "blocking_player_id"].value_counts(),
                     "hits": hits["hitting_player_id"].value_counts(),
                     "hits_taken": hits["hittee_player_id"].value_counts()}, axis=1)
    out.index = out.index.astype("int64")
    return out.fillna(0)


def _penalties(pbp: pd.DataFrame) -> pd.DataFrame:
    """Minor and major penalties taken and drawn, per player. Fighting majors are
    left out (both players get one) and counted as fights; misconducts are left out (no power play)."""
    pen = pbp[(pbp["event_type"] == "penalty") & pbp["details_type_code"].isin(["MIN", "MAJ"])
              & ~pbp["desc_key"].fillna("").str.startswith("fighting")]
    fights = pbp[(pbp["event_type"] == "penalty") & pbp["desc_key"].fillna("").str.startswith("fighting")]
    out = pd.concat({"penalties_taken": pen["committed_by_player_id"].value_counts(),
                     "penalties_drawn": pen["drawn_by_player_id"].value_counts(),
                     "fights": fights["committed_by_player_id"].value_counts()}, axis=1)
    out.index = out.index.astype("int64")
    return out.fillna(0)


def _onice_stats(ev: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.join(ev[["xg", "hd_xg", "is_goal", "rush", "cycle", "unblocked"]], on="ev")
    p["shots"] = 1
    cols = ["shots", "unblocked", "xg", "hd_xg", "is_goal", "rush", "cycle"]
    agg = p.groupby(["player_id", "side"])[cols].sum().unstack("side", fill_value=0)
    names = {"shots": "shots", "unblocked": "fenwick", "xg": "xg", "hd_xg": "hd_xg", "is_goal": "goals", "rush": "rush_shots", "cycle": "cycle_shots"}
    agg.columns = [f"{names[stat]}_{side}" for stat, side in agg.columns]
    agg = agg.rename(columns={"xg_for": "xgf", "xg_against": "xga", "hd_xg_against": "hd_xga"})
    return agg.drop(columns="hd_xg_for", errors="ignore")


def _on_off_ga(goals: pd.DataFrame, by_game: pd.DataFrame, shifts: pd.DataFrame, strength: str,
               on_ga: pd.Series, name: str = "ga_on_off_pct", weight: str | None = None) -> pd.Series:
    """Goals against per 60 with the player on the ice, as a % of his team's goals against
    per 60 with him off the ice, in the games he played (100 = no different, lower is better).
    ``goals`` are this strength's goals; ``on_ga`` each player's on-ice goals against. Any
    other event against works the same way (unblocked attempts for Fenwick against), and
    ``weight`` sums a column instead of counting events (``"xg"`` for xG against)."""
    secs = "toi_5v5" if strength == "5v5" else "toi_all"
    mine = by_game.groupby(["game_id", "team", "player_id"])[secs].sum().reset_index()
    if strength == "5v5":   # five skaters a side at 5v5, so team time is skater time / 5
        team_secs = mine.groupby(["game_id", "team"])[secs].sum() / 5
    else:                   # game length, from the last shift of each period
        length = shifts.groupby(["game_id", "period"])["end_seconds"].max().groupby("game_id").sum()
        team_secs = mine[["game_id", "team"]].drop_duplicates().set_index(["game_id", "team"])
        team_secs = team_secs.join(length.rename("s"), on="game_id")["s"]
    opp = goals["home_team"].where(goals["team"] != goals["home_team"], goals["away_team"])
    by = goals.groupby([goals["game_id"], opp.rename("team")])
    team_ga = by[weight].sum() if weight else by.size()
    m = mine.join(team_secs.rename("team_secs"), on=["game_id", "team"]).join(team_ga.rename("team_ga"), on=["game_id", "team"])
    tot = m.fillna({"team_ga": 0}).groupby("player_id")[[secs, "team_secs", "team_ga"]].sum()
    on = on_ga.reindex(tot.index).fillna(0)
    off_ga, off_secs = tot["team_ga"] - on, tot["team_secs"] - tot[secs]
    ratio = 100 * (on / tot[secs]) / (off_ga / off_secs)
    return ratio[(tot[secs] > 0) & (off_ga > 0) & (off_secs > 0)].rename(name)


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
    by_game = toi_by_game(shifts, skaters)
    times = toi(shifts, skaters, by_game)
    pairs = on_ice(ev, shifts)
    pairs = pairs[pairs["player_id"].isin(skaters)]

    gv_xga = _giveaway_xga(pbp, shots)
    frames = []
    pbp_5v5 = pbp["situation_code"].astype(str).str.zfill(4) == FIVE_ON_FIVE
    for strength, mask in (("all", slice(None)), ("5v5", ev["is_5v5"])):
        e = ev[mask] if strength == "5v5" else ev
        g = goals[goals["is_5v5"]] if strength == "5v5" else goals
        p = pairs[pairs["ev"].isin(e.index)]
        pb = pbp[pbp_5v5] if strength == "5v5" else pbp
        stats = (_individual(e, g).join(_onice_stats(e, p), how="outer").join(_zone_puck_events(pb), how="outer")
                 .join(_blocks_hits(pb), how="outer").join(_penalties(pb), how="outer").fillna(0))
        stats = stats[stats.index.isin(times.index)]
        stats = stats.join(gv_xga[strength]).fillna({"giveaway_xga": 0})
        stats = stats.join(_on_off_ga(g, by_game, shifts, strength, stats["goals_against"]))
        if strength == "5v5":   # in all situations these mostly measure power-play vs penalty-kill time
            stats = stats.join(_on_off_ga(e, by_game, shifts, strength, stats["xga"], "xga_on_off_pct", weight="xg"))
            stats = stats.join(_on_off_ga(e[e["unblocked"] == 1], by_game, shifts, strength,
                                          stats["fenwick_against"], "fa_on_off_pct"))
        # Goals per shot on goal; players without a shot on goal have no shooting %.
        stats["shooting_pct"] = 100 * stats["goals"] / stats["shots_on_goal"].where(stats["shots_on_goal"] > 0)
        # On-ice shares: his team's share of unblocked attempts (Fenwick %) and of xG while he's on the ice.
        ff = stats["fenwick_for"] + stats["fenwick_against"]
        stats["fenwick_pct"] = 100 * stats["fenwick_for"] / ff.where(ff > 0)
        xg = stats["xgf"] + stats["xga"]
        stats["xgf_pct"] = 100 * stats["xgf"] / xg.where(xg > 0)
        # Share of the opponents' shot attempts with him on the ice that he blocked.
        stats["block_share"] = 100 * stats["blocks"] / stats["shots_against"].where(stats["shots_against"] > 0)
        # Takeaways per giveaway; players without a giveaway have none.
        stats["take_give_ratio"] = stats["takeaways"] / stats["giveaways"].where(stats["giveaways"] > 0)
        long = stats.rename_axis("player_id").reset_index().melt("player_id", var_name="stat").dropna(subset=["value"])
        frames.append(long.assign(season=y, strength=strength))
    stats = pd.concat(frames, ignore_index=True)
    five = stats[(stats["strength"] == "5v5")].pivot(index="player_id", columns="stat", values="value")
    qoc = competition(shifts, skaters, five["xgf"], five["xga"])
    stats = pd.concat([stats, pd.DataFrame({"player_id": qoc.index.astype("int64"), "stat": qoc.name, "value": qoc.to_numpy(),
                                            "season": y, "strength": "5v5"})], ignore_index=True)

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
