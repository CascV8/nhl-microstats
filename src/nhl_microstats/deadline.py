"""Past seasons' team pages: each team's lineup after the trade deadline, read from the NHL's shift charts.

    nhl-microstats deadline-lineups 2023 2024 2025     # -> team_seasons/<year>.json, shipped with the code

Daily Faceoff keeps no history, so a complete season's lineup is the one each team settled on after the deadline:
over its games after TRADE_DEADLINE, every stretch of play in which nobody changed is credited to the players on the
ice, and the most-used combinations that share no player become the lineup:

    lines    forward trios at 5v5 (both teams five skaters and a goalie), the top four
    pairs    defence pairs at 5v5, the top three
    goalies  the two with the most starts (on the ice at the opening faceoff)
    pp       five-skater units on the power play (five against four), the top two
    pk       four-skater units on the penalty kill (four against five), the top two
    extras   everyone else who played for the team after the deadline

The opening-night lineup is read the same way from each team's first game alone (opening_lineups), so a team page
can show either.

Each team's final record, points percentage and special-teams percentages go in the same file (the NHL's season
summary), so a complete season's team pages need nothing from the network. Complete seasons never change; the file
is written once, where the season's games are kept, and committed.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import puckscrape as ps

from . import build, collect
from .teams import TEAM_STATS, _get, _key, _summary, NHL

log = logging.getLogger("nhl_microstats.deadline")

SHIPPED = Path(__file__).with_name("team_seasons")
TRADE_DEADLINE = {2021: dt.date(2022, 3, 21), 2022: dt.date(2023, 3, 3), 2023: dt.date(2024, 3, 8),
                  2024: dt.date(2025, 3, 7), 2025: dt.date(2026, 3, 6)}
FORWARD_ORDER = {"L": 0, "C": 1, "R": 2}


def _after_deadline(y: int) -> set[int]:
    """Regular-season game ids of season ``y`` played after its trade deadline."""
    s = ps.nhl.schedule(TRADE_DEADLINE[y] + dt.timedelta(days=1), dt.date(y + 1, 4, 30))
    return set(s.loc[(s["season"] == int(f"{y}{y + 1}")) & (s["game_type"] == 2), "game_id"].astype(int))


def _stretches(s: pd.DataFrame):
    """For one game-period's shifts: (seconds, player ids, their teams) for each stretch in which nobody changes."""
    start, end = s["start_seconds"].to_numpy(), s["end_seconds"].to_numpy()
    cuts = np.unique(np.concatenate([start, end]))
    a, b = cuts[:-1], cuts[1:]
    on = (start[:, None] <= a[None, :]) & (end[:, None] > a[None, :])
    ids, teams = s["player_id"].to_numpy(), s["team_abbrev"].to_numpy()
    for j in range(len(a)):
        yield int(b[j] - a[j]), ids[on[:, j]], teams[on[:, j]]


def _top(usage: Counter, n: int) -> list[tuple]:
    """The ``n`` most-used combinations that share no player."""
    taken, out = set(), []
    for combo, _ in usage.most_common():
        if not taken & set(combo):
            out.append(combo); taken |= set(combo)
            if len(out) == n:
                break
    return out


def _positions(y: int) -> dict[int, str]:
    """Skaters' positions (C, L, R, D); goalies aren't in the bios, which is how they're told apart."""
    return pd.read_parquet(collect.DATA / f"bios_{y}.parquet").set_index("player_id")["position"].to_dict()


def _usage(shifts: pd.DataFrame, pos: dict[int, str]):
    """Seconds each combination played, per (team, kind), and goalie starts per (team, goalie)."""
    shifts = shifts[shifts["end_seconds"] > shifts["start_seconds"]]
    usage: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for _, s in shifts.groupby(["game_id", "period"], sort=False):
        for secs, ids, teams in _stretches(s):
            sides = {t: ids[teams == t] for t in np.unique(teams)}
            if len(sides) != 2:
                continue
            (t1, p1), (t2, p2) = sides.items()
            for t, mine, theirs in ((t1, p1, p2), (t2, p2, p1)):
                sk = [int(p) for p in mine if p in pos]
                n_opp = sum(1 for p in theirs if p in pos)
                if len(mine) - len(sk) != 1 or len(theirs) - n_opp != 1:   # a goalie each side
                    continue
                if len(sk) == 5 and n_opp == 5:
                    f, d = sorted(p for p in sk if pos[p] != "D"), sorted(p for p in sk if pos[p] == "D")
                    if len(f) == 3 and len(d) == 2:
                        usage[t, "F"][tuple(f)] += secs; usage[t, "D"][tuple(d)] += secs
                elif len(sk) == 5 and n_opp == 4:
                    usage[t, "PP"][tuple(sorted(sk))] += secs
                elif len(sk) == 4 and n_opp == 5:
                    usage[t, "PK"][tuple(sorted(sk))] += secs
    starts = Counter(map(tuple, shifts.loc[(shifts["period"] == 1) & (shifts["start_seconds"] == 0)
                                           & ~shifts["player_id"].isin(list(pos)), ["team_abbrev", "player_id"]]
                         .astype({"player_id": "int64"}).itertuples(index=False, name=None)))
    return usage, starts


def _lineup(abbr: str, usage, starts, pos: dict[int, str]) -> dict:
    """The team's most-used combinations that share no player, and its two goalies with the most starts."""
    goalies = [p for (team, p), _ in sorted(starts.items(), key=lambda kv: -kv[1]) if team == abbr][:2]
    return {"lines": [sorted(c, key=lambda p: FORWARD_ORDER.get(pos.get(p), 1)) for c in _top(usage[abbr, "F"], 4)],
            "pairs": [list(c) for c in _top(usage[abbr, "D"], 3)], "goalies": goalies,
            "pp": [list(c) for c in _top(usage[abbr, "PP"], 2)], "pk": [list(c) for c in _top(usage[abbr, "PK"], 2)]}


def _first_games(y: int) -> dict[str, tuple[int, str, tuple[str, str]]]:
    """Each team's first regular-season game of season ``y``: team -> (game id, date, (away, home))."""
    s = ps.nhl.schedule(dt.date(y, 9, 15), dt.date(y, 11, 15))
    s = s[(s["season"] == int(f"{y}{y + 1}")) & (s["game_type"] == 2)].sort_values(["date", "game_id"])
    first = {}
    for g in s.itertuples():
        for team in (g.away_team, g.home_team):
            first.setdefault(team, (int(g.game_id), str(g.date), (g.away_team, g.home_team)))
    return first


def opening_lineups(y: int) -> dict[str, dict]:
    """Each team's lineup on opening night: the combinations it used in its first game. Reads only those games, so
    it also runs where only the season under way's games are kept; a team whose first game isn't here is left out."""
    pos, out = _positions(y), {}
    for team, (gid, date, sides) in _first_games(y).items():
        path = collect.game_dir(y) / f"{gid}_shifts.parquet"
        if not path.exists():
            continue
        # as collect.load_games cleans them; and only this game's two teams (a chart can carry another game's shifts)
        s = (pd.read_parquet(path).assign(game_id=gid).dropna(subset=["player_id"]).astype(collect.SHIFT_KEEP)
             .sort_values("end_seconds").drop_duplicates(["game_id", "player_id", "period", "start_seconds"], keep="last"))
        usage, starts = _usage(s[s["team_abbrev"].isin(sides)], pos)
        out[team] = {"game_id": gid, "date": date} | _lineup(team, usage, starts, pos)
    return out


def deadline_lineups(y: int) -> dict:
    """Season ``y``'s teams: final record and stats, the lineup after the deadline and the opening-night lineup (see
    the module docstring)."""
    _, shifts = build._load(y)
    shifts = shifts[shifts["game_id"].isin(_after_deadline(y))]
    pos = _positions(y)
    usage, starts = _usage(shifts, pos)
    played = shifts.groupby("team_abbrev")["player_id"].agg(lambda p: sorted(set(p.astype("int64"))))
    opening = opening_lineups(y)

    # names and final numbers: the NHL's standings late in the season (abbreviation and name) and its season summary
    table = _get(f"{NHL}/standings/{y + 1}-04-01").json()["standings"]
    summary = {_key(t["teamFullName"]): t for t in _summary(y)}
    out = {"season": y, "deadline": TRADE_DEADLINE[y].isoformat(), "teams": []}
    for t in table:
        abbr, name = t["teamAbbrev"]["default"], t["teamName"]["default"]
        s = summary.get(_key(name)) or {}
        after = _lineup(abbr, usage, starts, pos)
        dressed = {p for c in after["lines"] + after["pairs"] for p in c} | set(after["goalies"])
        out["teams"].append({
            "abbrev": abbr, "name": name, "gp": s.get("gamesPlayed"), "points": s.get("points"),
            "record": f'{s.get("wins")}-{s.get("losses")}-{s.get("otLosses")}',
            "stats": {k: s.get(k) for k, _ in TEAM_STATS},
            "games_after": int(shifts.loc[shifts["team_abbrev"] == abbr, "game_id"].nunique()),
            **after, "extras": [p for p in played.get(abbr, []) if p not in dressed],
            "opening": opening.get(abbr),
        })
    return out


def write(y: int) -> Path:
    SHIPPED.mkdir(exist_ok=True)
    path = SHIPPED / f"{y}.json"
    path.write_text(json.dumps(deadline_lineups(y)), encoding="utf-8")
    log.info("%s: deadline lineups written to %s", y, path)
    return path
