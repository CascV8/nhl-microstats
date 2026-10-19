"""Puck possession stats from the NHL's goal tracking.

The NHL publishes puck and player tracking for every goal: the ~14 s before it,
10 frames a second, x/y for the puck and all skaters (the data behind the goal
replays on nhl.com). It covers goals only, so these stats describe how goals
were built, not every possession.

    collect_goal_tracking(2025)   # download -> data/goal_tracking/<season>/<game>.parquet
    build_goal_tracking(2025)     # stats -> microstats.db, source 'goal-tracking'

Per player, from the scoring team's side of each goal:
  gt_possession_time   seconds with the puck in the build-up
  gt_carries           zone entries carrying the puck over the offensive blue line
  gt_defenders_beaten  opposing skaters who were ahead of the carrier, close to
                       him, and behind him by the time he gave up the puck
  gt_dekes             the puck swung across the carrier's body with an opponent close
and from the conceding side:
  gt_beaten            times he was one of those beaten defenders (fewer is better)
(Passes are left out: primary assists and their xG already cover them.)
"""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from puckscrape._util import start_year

from .collect import DATA, game_dir

log = logging.getLogger("nhl_microstats.goal_tracking")

URL = "https://wsr.nhle.com/sprites/{season}/{game}/ev{event}.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nhl.com/"}   # the replay viewer's request
FPS = 10

# Rink in tracking units (inches): x 0-2400 along the ice, y 0-1020 across it.
LENGTH, WIDTH = 2400, 1020
GOAL_LINE = 132            # 11 ft from the end boards
BLUE_LINE = 900            # 75 ft from the end boards
# Possession: the nearest skater within 5 ft of the puck, with the puck moving
# along with him (a pass or shot travels far faster than a skater).
REACH = 60
NET_HALF, NET_DEPTH = 40, 44   # inside the net's mouth (6 ft wide) and depth, with a little slack
CARRY_SPEED = 480          # in/s (40 ft/s) of puck speed relative to the skater; passes and shots go faster
MIN_FRAMES = 2             # a single frame is noise (a puck going past); catch-and-release shots take 0.2-0.3 s
GAP_FRAMES = 5             # drop-outs of up to 0.5 s inside one possession are bridged
CONTEST = 120              # a defender within 10 ft of the carrier is contesting him
# Deke: the puck swings across the carrier's body (2.5 ft or more, from one side to the
# other) within 0.5 s, with an opponent or the goalie within 10 ft. Possessions under 0.5 s can't show one.
DEKE_SWING, DEKE_FRAMES, DEKE_MIN_POSSESSION = 30, 5, 5

STATS = {
    # stat -> (label, per60, higher is better)
    "gt_possession_time":  ("Puck possession on goals (s)", True, True),
    "gt_carries":          ("Zone carries on goals", True, True),
    "gt_defenders_beaten": ("Defenders beaten on goals", True, True),
    "gt_dekes":            ("Dekes on goals", True, True),
    "gt_beaten":           ("Beaten on goals against", True, False),
}


def tracking_dir(season):
    return DATA / "goal_tracking" / str(start_year(season))


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _frames(js: list, game_id: int, event_id: int) -> pd.DataFrame:
    t0 = js[0]["timeStamp"]
    rows = [(game_id, event_id, (f["timeStamp"] - t0) / FPS, o.get("playerId") or 0, o.get("teamAbbrev") or "",
             o["x"], o["y"]) for f in js for o in f["onIce"].values() if "x" in o and "y" in o]   # some frames lose an object
    return pd.DataFrame(rows, columns=["game_id", "event_id", "t", "player_id", "team", "x", "y"])


def _fetch_game(session: requests.Session, game_id: int, goals: pd.DataFrame, season: int, out) -> int:
    frames = []
    for e in goals["event_id"]:
        r = session.get(URL.format(season=f"{season}{season + 1}", game=game_id, event=e), headers=HEADERS, timeout=30)
        js = r.json() if r.status_code == 200 and r.content.strip() else None
        if js:   # some goals have no tracking (an empty file or a 403)
            frames.append(_frames(js, game_id, int(e)))
    df = pd.concat(frames, ignore_index=True) if frames else _frames([{"timeStamp": 0, "onIce": {}}], game_id, 0)
    df.astype({"player_id": "int64", "x": "float32", "y": "float32", "t": "float32"}).to_parquet(out)
    return len(frames)


def collect_goal_tracking(season, workers: int = 6) -> None:
    y = start_year(season)
    folder = tracking_dir(y)
    folder.mkdir(parents=True, exist_ok=True)
    todo = []
    for path in sorted(game_dir(y).glob("*_pbp.parquet")):
        game_id = int(path.name.split("_")[0])
        out = folder / f"{game_id}.parquet"
        if out.exists():
            continue
        pbp = pd.read_parquet(path, columns=["event_id", "event_type", "game_type", "period_type"])
        goals = pbp[(pbp["event_type"] == "goal") & (pbp["game_type"] == 2) & (pbp["period_type"] != "SO")]
        todo.append((game_id, goals, out))
    log.info("%s: %d games to download", y, len(todo))
    session = requests.Session()
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_fetch_game, session, g, goals, y, out): g for g, goals, out in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:   # one bad game shouldn't stop the season; it's retried next run
                log.warning("game %s failed: %s", futs[f], e)
            done += 1
            if done % 100 == 0:
                log.info("%s: %d / %d games", y, done, len(todo))


# --------------------------------------------------------------------------
# Possessions
# --------------------------------------------------------------------------

def trim(goal: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Cut a goal's clip at the moment the puck enters the net (clips run a few
    seconds past it, when players fish the puck out), and say which way the
    scoring team attacked: True if toward x = 0."""
    puck = goal[goal["player_id"] == 0].drop_duplicates("t").sort_values("t")
    depth = np.maximum(GOAL_LINE - puck["x"], puck["x"] - (LENGTH - GOAL_LINE))   # how far past a goal line
    in_net = (depth > 0) & (depth <= NET_DEPTH) & ((puck["y"] - WIDTH / 2).abs() <= NET_HALF)
    if in_net.any():
        first = puck[in_net].iloc[0]
        return goal[goal["t"] <= first["t"]], bool(first["x"] < LENGTH / 2)
    return goal, bool(not puck.empty and puck["x"].iloc[-1] < LENGTH / 2)


def possessions(goal: pd.DataFrame, skaters: set[int]) -> pd.DataFrame:
    """Possession segments in one (trimmed) goal clip: player, team, first and last frame."""
    puck = goal[goal["player_id"] == 0].drop_duplicates("t").set_index("t")[["x", "y"]]
    sk = goal[goal["player_id"].isin(skaters)]
    if puck.empty or sk.empty:
        return pd.DataFrame(columns=["player_id", "team", "start", "end"])
    px = sk.pivot_table(index="t", columns="player_id", values="x").reindex(puck.index)
    py = sk.pivot_table(index="t", columns="player_id", values="y").reindex(puck.index)
    dist = np.hypot(px.sub(puck["x"], axis=0), py.sub(puck["y"], axis=0))
    # Puck speed relative to each skater, in inches per second.
    rel = np.hypot(px.diff().sub(puck["x"].diff(), axis=0), py.diff().sub(puck["y"].diff(), axis=0)) * FPS
    d = dist.to_numpy(dtype=float, na_value=np.inf)        # frames can lose a skater, or all of them
    near = d.argmin(axis=1)
    rows = np.arange(len(d))
    r = rel.to_numpy(dtype=float, na_value=0.0)[rows, near]   # first frame has no speed yet
    ok = (d[rows, near] <= REACH) & (r <= CARRY_SPEED)
    holder = pd.Series(np.where(ok, dist.columns.to_numpy()[near], 0), index=puck.index, dtype="int64")
    team = sk.drop_duplicates("player_id").set_index("player_id")["team"]
    segs, cur = [], None
    for i, p in enumerate(holder.to_numpy()):
        if p == 0:
            continue
        if cur and cur[0] == p and i - cur[2] <= GAP_FRAMES + 1:
            cur[2] = i
        else:
            if cur:
                segs.append(cur)
            cur = [p, i, i]
    if cur:
        segs.append(cur)
    segs = [s for s in segs if s[2] - s[1] + 1 >= MIN_FRAMES]
    # A blip of someone else's "possession" (the puck passing close by) can split one carry; rejoin it.
    merged = []
    for s in segs:
        if merged and merged[-1][0] == s[0] and s[1] - merged[-1][2] <= GAP_FRAMES + 1:
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    segs = merged
    out = pd.DataFrame(segs, columns=["player_id", "start", "end"])
    out["team"] = out["player_id"].map(team)
    return out


def _dekes(puck: np.ndarray, carrier: np.ndarray, pressured: np.ndarray, toward_net: float) -> int:
    """Dekes in one possession. ``puck`` and ``carrier`` are (frames, 2) positions;
    ``pressured`` says whether an opponent is within CONTEST of the carrier."""
    v = np.gradient(carrier, axis=0)
    speed = np.hypot(v[:, 0], v[:, 1])
    moving = speed > 1   # an inch a frame; standing still, use the direction of the net
    hx = np.where(moving, v[:, 0] / np.where(moving, speed, 1), toward_net)
    hy = np.where(moving, v[:, 1] / np.where(moving, speed, 1), 0.0)
    rel = puck - carrier
    side = rel[:, 0] * -hy + rel[:, 1] * hx   # puck offset across the body, + = his left
    n, i = 0, 0
    while i < len(side):
        for j in range(i + 1, min(len(side), i + DEKE_FRAMES + 1)):
            if (np.sign(side[i]) != np.sign(side[j]) and abs(side[j] - side[i]) >= DEKE_SWING
                    and pressured[j]):
                n += 1
                i = j
                break
        else:
            i += 1
    return n


def goal_stats(goal: pd.DataFrame, scorer_team: str, skaters: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stats for the scoring team's skaters in one goal, and the possessions they came from."""
    goal, flip = trim(goal)
    segs = possessions(goal, skaters)
    puck = goal[goal["player_id"] == 0].drop_duplicates("t").reset_index(drop=True)
    cols = list(STATS)
    if segs.empty or puck.empty:
        return pd.DataFrame(columns=cols), segs
    # Flip so the attacked goal is at x = LENGTH.
    def fx(x):
        return LENGTH - x if flip else x
    pxs, pys = fx(puck["x"].to_numpy()), puck["y"].to_numpy()
    sk = goal[goal["player_id"].isin(skaters)]
    times = puck["t"].to_numpy()
    pos = {p: g.drop_duplicates("t").set_index("t").reindex(times)[["x", "y"]].to_numpy().copy()
           for p, g in sk.groupby("player_id")}
    for p in pos:
        pos[p][:, 0] = fx(pos[p][:, 0])
    team_of = sk.drop_duplicates("player_id").set_index("player_id")["team"]
    opponents = [p for p in pos if team_of[p] != scorer_team]
    # Every opponent including the goalie (not in ``skaters``), for deke pressure.
    them = goal[(goal["player_id"] != 0) & (goal["team"] != scorer_team)].drop_duplicates(["t", "player_id"])
    opp_x = fx(them.pivot(index="t", columns="player_id", values="x").reindex(times).to_numpy(dtype=float))
    opp_y = them.pivot(index="t", columns="player_id", values="y").reindex(times).to_numpy(dtype=float)
    puck_xy = np.column_stack([pxs, pys])

    rows = {}
    def add(p, stat, n=1):
        rows.setdefault(p, dict.fromkeys(cols, 0.0))[stat] += n

    att = segs[segs["team"] == scorer_team]
    for s in att.itertuples():
        p, a, b = s.player_id, s.start, s.end
        add(p, "gt_possession_time", (b - a + 1) / FPS)
        if pxs[a] < LENGTH - BLUE_LINE <= pxs[b]:
            add(p, "gt_carries")
        cx = pos[p][:, 0]
        cy = pos[p][:, 1]
        for d in opponents:
            dx, dy = pos[d][:, 0], pos[d][:, 1]
            gap = np.hypot(dx[a:b + 1] - cx[a:b + 1], dy[a:b + 1] - cy[a:b + 1])
            if np.isnan(gap).all() or np.nanmin(gap) > CONTEST:
                continue
            if dx[a] > cx[a] and dx[b] < cx[b]:   # ahead of the carrier, then behind him
                add(p, "gt_defenders_beaten")
                add(d, "gt_beaten")
        if b - a + 1 >= DEKE_MIN_POSSESSION and opp_x.shape[1]:
            near = np.hypot(opp_x[a:b + 1] - cx[a:b + 1, None], opp_y[a:b + 1] - cy[a:b + 1, None])
            pressured = (np.nan_to_num(near, nan=np.inf) <= CONTEST).any(axis=1)
            add(p, "gt_dekes", _dekes(puck_xy[a:b + 1], pos[p][a:b + 1], pressured, 1.0))
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols), segs


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_goal_tracking(season) -> pd.DataFrame:
    from .db import DB_PATH, init
    y = start_year(season)
    init()
    with sqlite3.connect(DB_PATH) as con:
        skaters = set(pd.read_sql("SELECT player_id FROM player_seasons WHERE season = ?", con, params=[y])["player_id"])
    frames = {"5v5": [], "all": []}
    checks = []
    for path in sorted(tracking_dir(y).glob("*.parquet")):
        trk = pd.read_parquet(path)
        game_id = int(path.stem)
        pbp = pd.read_parquet(game_dir(y) / f"{game_id}_pbp.parquet")
        goals = pbp[pbp["event_type"] == "goal"].set_index("event_id")
        for e, goal in trk.groupby("event_id"):
            if e not in goals.index:
                continue
            g = goals.loc[e]
            st, segs = goal_stats(goal, g["event_team"], skaters)
            if st.empty:
                continue
            frames["all"].append(st)
            if str(g["situation_code"]) == "1551":
                frames["5v5"].append(st)
            # Sanity check against the official scoring (tips and deflections are too quick to count as possession).
            ids = segs.loc[segs["team"] == g["event_team"], "player_id"].tolist()
            others = [i for i in ids if i != g["scoring_player_id"]]
            a1 = g["assist1_player_id"]
            checks.append((str(g["shot_type"]) in ("tip-in", "deflected"), bool(ids) and ids[-1] == g["scoring_player_id"],
                           pd.notna(a1), pd.notna(a1) and bool(others) and others[-1] == a1))
    c = pd.DataFrame(checks, columns=["tip", "scorer", "has_a1", "a1"])
    log.info("%s: %d goals tracked. Last possession is the scorer on %.0f%% of non-tip goals; "
             "the last other teammate is the primary assist on %.0f%% of assisted goals",
             y, len(c), 100 * c.loc[~c["tip"], "scorer"].mean(), 100 * c.loc[c["has_a1"], "a1"].mean())
    rows = []
    for strength, fs in frames.items():
        # Every skater was on the ice for goals; one who never touched the puck in a build-up has zeros.
        tot = pd.concat(fs).groupby(level=0).sum().reindex(sorted(skaters), fill_value=0.0)
        long = tot.rename_axis("player_id").reset_index().melt("player_id", var_name="stat")
        rows.append(long.assign(season=y, strength=strength))
    out = pd.concat(rows)[["player_id", "season", "strength", "stat", "value"]]
    with sqlite3.connect(DB_PATH) as con:
        con.execute(f"DELETE FROM stats WHERE season = ? AND stat IN ({','.join('?' * len(STATS))})", (y, *STATS))
        con.executemany("INSERT INTO stats VALUES (?, ?, ?, ?, ?)", out.itertuples(index=False))
    from .curves import fit_all
    fit_all(list(STATS))
    return out
