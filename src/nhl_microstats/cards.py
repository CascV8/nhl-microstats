"""Player cards: a self-contained HTML page of microstat percentiles.

    nhl-microstats cards                      # top 12 forwards + top 4 D by points
    nhl-microstats cards McDavid Makar Hutson --out my_cards.html
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from puckscrape._util import start_year

from . import collect, curves
from .db import connect
from .percentile import MIN_TOI, REFERENCE, _seasons, complete_seasons, player_table, season_name

TEMPLATE = Path(__file__).with_name("cards_template.html")

# Stats shown on a card, by section.
SECTIONS = {
    "Shooting": ["goals", "ixg", "shots_on_goal", "shooting_pct", "rush_shots", "cycle_shots"],
    "Playmaking": ["primary_assists", "secondary_assists", "primary_assist_xg"],
    "Puck battles": ["oz_takeaways", "dz_takeaways", "oz_giveaways", "dz_giveaways"],
    "On-ice offence": ["shots_for", "xgf", "rush_shots_for", "cycle_shots_for"],
    "On-ice defence": ["shots_against", "xga", "rush_shots_against", "cycle_shots_against"],
    "NHL Edge": ["max_skating_speed", "bursts_over_20mph", "shot_strength", "oz_time_pct"],
}
# Opt-in section (`cards --goal-tracking`): possession stats from the NHL's goal tracking, goals only.
GOAL_TRACKING = {"Goal tracking (goals only)": ["gt_possession_time", "gt_carries", "gt_defenders_beaten", "gt_dekes"]}
MODES = {"5v5": ("5v5", True), "all": ("all", True)}   # mode -> (strength, per60)


def top_players(season, forwards: int = 12, defense: int = 4) -> list[int]:
    y = start_year(season)
    with connect() as con:
        d = pd.read_sql("""SELECT p.player_id, p.pos_group, SUM(s.value) AS pts FROM stats s
                           JOIN player_seasons p USING (player_id, season)
                           WHERE s.season = ? AND s.strength = 'all'
                             AND s.stat IN ('goals', 'primary_assists', 'secondary_assists')
                           GROUP BY p.player_id""", con, params=[y])
    d = d.sort_values("pts", ascending=False)
    return (d[d["pos_group"] == "F"].head(forwards)["player_id"].tolist()
            + d[d["pos_group"] == "D"].head(defense)["player_id"].tolist())


def _xgf_share(con: sqlite3.Connection, y: int, vs: list[int]) -> pd.DataFrame:
    """5v5 xGF% for every skater in season ``y``, with a percentile among
    qualified player-seasons at their position in the ``vs`` seasons."""
    d, ref = _shares(con, [y]), _shares(con, vs)
    pool = ref[ref["toi_5v5"] >= MIN_TOI["5v5"] * 60]
    fits = {g: curves.fit(pool.loc[pool["pos_group"] == g, "share"]) for g in ("F", "D")}
    d["pct"] = [fits[g](v) for g, v in zip(d["pos_group"], d["share"])]
    return d.set_index("player_id")


def _shares(con: sqlite3.Connection, seasons: list[int]) -> pd.DataFrame:
    d = pd.read_sql(f"""SELECT p.player_id, p.season, p.pos_group, p.toi_5v5,
                               SUM(CASE WHEN s.stat = 'xgf' THEN s.value END) AS xgf,
                               SUM(CASE WHEN s.stat = 'xga' THEN s.value END) AS xga
                        FROM stats s JOIN player_seasons p USING (player_id, season)
                        WHERE s.season IN ({','.join('?' * len(seasons))}) AND s.strength = '5v5'
                        GROUP BY p.player_id, p.season""", con, params=list(seasons))
    d["share"] = 100 * d["xgf"] / (d["xgf"] + d["xga"])
    return d


def _games_played(y: int) -> dict[int, int]:
    """Games played per player from NHL Edge, which counts every game -- including
    the ones without shift charts that our TOI-based GP leaves out."""
    path = collect.DATA / f"edge_{y}.parquet"
    if not path.exists():
        return {}
    e = pd.read_parquet(path, columns=["player_id", "info_games_played"]).dropna()
    return dict(zip(e["player_id"].astype(int), e["info_games_played"].astype(int)))


def card_data(players, season="2025", vs_season=REFERENCE, goal_tracking: bool = False) -> dict:
    """Card data for ``players`` in ``season``, ranked against the ``vs_season``
    pool (default: the last three complete seasons)."""
    y = start_year(season)
    vs = _seasons(vs_season)
    sections = SECTIONS | (GOAL_TRACKING if goal_tracking else {})
    with connect() as con:
        share = _xgf_share(con, y, vs)
    # Mid-season, games without shift charts yet can leave GP short of games played; say so.
    played = _games_played(y) if y not in complete_seasons() else {}
    cards = []
    for p in players:
        modes = {}
        for mode, (strength, per60) in MODES.items():
            t = player_table(p, y, strength, per60, pool_season=vs).set_index("stat")
            info = t.attrs["player"]
            modes[mode] = {stat: {"v": round(float(t.loc[stat, "value"]), 3), "pct": round(float(t.loc[stat, "percentile"]), 1)}
                           for stats in sections.values() for stat in stats if stat in t.index}
        totals = player_table(p, y, "all", per60=False, pool_season=vs).set_index("stat")
        labels = totals["label"]
        pid = int(info["player_id"])
        cards.append({
            "id": pid, "name": info["name"], "position": info["position"], "group": info["pos_group"],
            "team": info["team"], "gp": int(info["gp"]),
            "gp_played": max(played.get(int(info["player_id"]), 0), int(info["gp"])),
            "toi": round(info["toi_all"] / 60 / max(info["gp"], 1), 1),
            "g": int(totals.loc["goals", "value"]),
            "a": int(totals.loc["primary_assists", "value"] + totals.loc["secondary_assists", "value"]),
            "xgf_share": round(float(share.loc[pid, "share"]), 1), "xgf_share_pct": round(float(share.loc[pid, "pct"]), 1),
            "modes": modes,
        })
    return {"season": season_name([y]), "vs_season": season_name(vs), "early": y not in vs and y > max(vs),
            "sections": sections,
            "labels": {s: labels.get(s, s) for stats in sections.values() for s in stats},
            "lower_is_better": ["shots_against", "xga", "rush_shots_against", "cycle_shots_against", "oz_giveaways", "dz_giveaways"],
            "min_toi": MIN_TOI, "cards": cards}


def write_cards(players=None, season="2025", out="cards.html", vs_season=REFERENCE, goal_tracking: bool = False) -> Path:
    data = card_data(players or top_players(season), season, vs_season, goal_tracking)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    if len(data["cards"]) == 1:
        html = html.replace("<title>NHL Microstat Cards</title>", f"<title>{data['cards'][0]['name']} Card</title>", 1)
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {len(data['cards'])} cards to {out.resolve()}")
    return out
