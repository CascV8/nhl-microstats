"""Player cards: a self-contained HTML page of microstat percentiles.

    nhl-microstats cards                      # top 12 forwards + top 4 D by points
    nhl-microstats cards McDavid Makar Hutson --out my_cards.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from puckscrape._util import start_year

from . import collect, curves
from .db import connect
from .percentile import (MIN_TOI, REFERENCE, SEASON_RELATIVE, _defs, _seasons, complete_seasons, curve_for, pool,
                         season_name, season_table)

TEMPLATE = Path(__file__).with_name("cards_template.html")

# Stats shown on a card, by section. Stats not yet given a section sit under "" (no heading).
SECTIONS = {
    "Finishing": ["goals", "shooting_pct", "shot_strength"],
    "Play driving": ["primary_assists", "secondary_assists", "primary_assist_xg", "gt_possession_time", "gt_carries"],
    "Offence": ["rush_shots", "shots_on_goal", "fenwick_pct", "cycle_shots", "oz_time_pct", "gt_dekes", "gt_defenders_beaten", "ixg"],
    "Defence": ["take_give_ratio", "blocks", "block_share", "ga_on_off_pct", "fa_on_off_pct", "xgf_pct_rel", "gt_beaten"],
    "Penalties": ["penalties_taken", "penalties_drawn"],
    "Speed": ["bursts_over_20mph", "max_skating_speed"],
    "Competition": ["qoc_xgf_pct"],
    "Physicality": ["hits", "hits_taken", "fights"],
}
# gt_* stats come from the NHL's goal tracking (2023-24 on) and are missing before then.
# The headline number at the top of each card, in the current mode.
HEADLINE = "xgf_pct"
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


def _games_played(y: int) -> dict[int, int]:
    """Games played per player from NHL Edge, which counts every game -- including
    the ones without shift charts that our TOI-based GP leaves out."""
    path = collect.DATA / f"edge_{y}.parquet"
    if not path.exists():
        return {}
    e = pd.read_parquet(path, columns=["player_id", "info_games_played"]).dropna()
    return dict(zip(e["player_id"].astype(int), e["info_games_played"].astype(int)))


def category_curves(vs: list[int], strength: str) -> dict[str, dict[str, curves.Curve]]:
    """A curve per position group and section that turns a category average into a percentile.

    Every qualified player-season in the ``vs`` pool gets the same category averages a card
    shows (the mean of his stat percentiles in that section), and the averages are fitted
    with the same Yeo-Johnson + logistic curve as single stats. Forwards and defensemen
    get separate curves. An average of percentiles bunches toward 50; this spreads it
    back out, so a 70 average that only 5% of players reach comes out near the 95th."""
    stats = sorted({s for ss in SECTIONS.values() for s in ss})
    defs = _defs()
    marks = ",".join("?" * len(vs))
    with connect() as con:
        ps = pd.read_sql(f"SELECT player_id, season, pos_group, toi_all, toi_5v5 FROM player_seasons WHERE season IN ({marks})",
                         con, params=vs).set_index(["player_id", "season"])
        st = pd.read_sql(f"SELECT player_id, season, strength, stat, value FROM stats WHERE season IN ({marks}) "
                         f"AND stat IN ({','.join('?' * len(stats))})", con, params=[*vs, *stats])
    ps = ps[ps[f"toi_{strength}"] >= MIN_TOI[strength] * 60]
    out = {}
    for pos in curves.POSITIONS:
        members = ps[ps["pos_group"] == pos]
        pcts = {}
        for stat in stats:
            s = st[st["stat"] == stat]
            if s.empty or stat not in defs.index:
                continue
            # The same strength the card uses: this one if the stat has it (Edge is all-situations only).
            have = s["strength"].unique()
            st_strength = strength if strength in have else have[0]
            v = s[s["strength"] == st_strength].set_index(["player_id", "season"])["value"].reindex(members.index)
            # Season-relative stats (takeaways, giveaways) rank each season against itself, as on the card.
            parts = [(y, [y]) for y in vs] if stat in SEASON_RELATIVE else [(None, vs)]
            ranked = []
            for y, spec in parts:
                values, info = pool(stat, spec, pos, st_strength, True)
                if len(values) < curves.MIN_POOL:
                    continue
                vy = v if y is None else v[v.index.get_level_values("season") == y]
                if info["per60"]:
                    vy = vy / members.loc[vy.index, f"toi_{st_strength}"] * 3600
                vy = vy[np.isfinite(vy)]
                p = pd.Series(curve_for(values, info)(vy.to_numpy()), index=vy.index)
                ranked.append(p if info["higher_is_better"] else 100 - p)
            if ranked:
                pcts[stat] = pd.concat(ranked)
        p = pd.DataFrame(pcts).reindex(members.index)
        out[pos] = {}
        for title, ss in SECTIONS.items():
            avg = p[[s for s in ss if s in p]].mean(axis=1).dropna()
            if len(avg) >= curves.MIN_POOL and np.ptp(avg) > 0:
                out[pos][title] = curves.fit(avg)
    return out


def _find(ps_: pd.DataFrame, player, y: int) -> int:
    """A player id from an id or (part of) a name, among one season's skaters."""
    if str(player).isdigit() and int(player) in ps_.index:
        return int(player)
    match = ps_[ps_["name"].str.contains(str(player), case=False, regex=False)]
    if match.empty:
        raise LookupError(f"no skater matching {player!r} in {y}-{(y + 1) % 100}")
    if len(match) > 1:
        raise LookupError(f"{len(match)} skaters match {player!r}: {', '.join(match['name'].head(8))}")
    return int(match.index[0])


def card_data(players=None, season="2025", vs_season=REFERENCE) -> dict:
    """Card data for ``players`` (default: every skater) in ``season``, ranked against
    the ``vs_season`` pool (default: the last three complete seasons)."""
    y = start_year(season)
    vs = _seasons(vs_season)
    # Mid-season, games without shift charts yet can leave GP short of games played; say so.
    played = _games_played(y) if y not in complete_seasons() else {}
    cat_curves = {mode: category_curves(vs, strength) for mode, (strength, _) in MODES.items()}
    shown = [*{s for ss in SECTIONS.values() for s in ss}, HEADLINE]
    tables = {}
    for mode, (strength, per60) in MODES.items():
        t = season_table(y, strength, per60, pool_season=vs)
        tables[mode] = t[t["stat"].isin(shown)]
    ps_ = t.attrs["players"]
    labels = dict(zip(t["stat"], t["label"]))
    ids = [_find(ps_, p, y) for p in players] if players else list(ps_.index)
    with connect() as con:
        tot = pd.read_sql("SELECT player_id, stat, value FROM stats WHERE season = ? AND strength = 'all' "
                          "AND stat IN ('goals', 'primary_assists', 'secondary_assists')", con, params=[y])
    tot = tot.pivot(index="player_id", columns="stat", values="value").reindex(ids).fillna(0)
    by_player = {mode: {pid: {r.stat: {"v": round(float(r.value), 3), "pct": round(float(r.percentile), 1)}
                              for r in g.itertuples()}
                        for pid, g in t[t["player_id"].isin(ids)].groupby("player_id")}
                 for mode, t in tables.items()}
    cards = []
    for pid in ids:
        info = ps_.loc[pid]
        modes = {mode: by_player[mode].get(pid, {}) for mode in MODES}
        cards.append({
            "id": int(pid), "name": info["name"], "position": info["position"], "group": info["pos_group"],
            "team": info["team"], "gp": int(info["gp"]),
            "gp_played": max(played.get(int(pid), 0), int(info["gp"])),
            "toi": round(info["toi_all"] / 60 / max(info["gp"], 1), 1),
            "g": int(tot.loc[pid, "goals"]),
            "a": int(tot.loc[pid, "primary_assists"] + tot.loc[pid, "secondary_assists"]),
            "modes": modes,
            # Per mode: [section, average of its stat percentiles, number of stats, percentile of that average].
            "categories": {mode: _categories(m, cat_curves[mode][info["pos_group"]]) for mode, m in modes.items()},
        })
    # Only sections with data for this season (no goal tracking before 2023-24).
    sections = {t: ss for t, ss in SECTIONS.items() if any(s in c["modes"][m] for c in cards for m in MODES for s in ss)}
    return {"season": season_name([y]), "vs_season": season_name(vs), "early": y not in vs and y > max(vs),
            "sections": sections,
            "labels": {s: labels.get(s, s) for stats in sections.values() for s in stats}, "headline": HEADLINE,
            "min_toi": MIN_TOI, "cards": cards}


def _categories(m: dict, fitted: dict[str, curves.Curve]) -> list:
    rows = []
    for title, ss in SECTIONS.items():
        ps = [m[s]["pct"] for s in ss if s in m and np.isfinite(m[s]["pct"])]
        if ps and (title in fitted or len(ps) == 1):
            avg = float(np.mean(ps))
            # One stat: its percentile already is the category's percentile.
            pct = avg if len(ps) == 1 else fitted[title](avg)
            rows.append([title, round(avg, 1), len(ps), round(pct, 1)])
    return rows


def write_cards(players=None, season="2025", out="cards.html", vs_season=REFERENCE) -> Path:
    data = card_data(players or top_players(season), season, vs_season)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    if len(data["cards"]) == 1:
        html = html.replace("<title>NHL Microstat Cards</title>", f"<title>{data['cards'][0]['name']} Card</title>", 1)
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {len(data['cards'])} cards to {out.resolve()}")
    return out
