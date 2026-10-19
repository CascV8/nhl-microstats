"""Value in, percentile out.

The comparison pool for a stat is every qualified skater (enough TOI at that
strength) in the chosen season and position group: forwards and defensemen
each have their own pool and curve, never a mixed one. Values are rates per 60
minutes by default (Edge speeds, shot speed and zone-time shares are used as
they are). The percentile comes from a logistic curve fitted to the pool (see
curves.py); the raw mid-rank is returned alongside as ``empirical_percentile``.
For stats where less is better (giveaways, "against" stats, DZ time) both are
flipped, so a higher percentile is always better.

    >>> percentile("rush_shots_for", 3.5, season=2025, position="F")
    {'percentile': 57.2, ...}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from puckscrape._util import start_year

from . import curves
from .db import connect

MIN_TOI = {"5v5": 200, "all": 300}   # minutes to qualify for the pool
REFERENCE = "last3"                  # default pool: every qualified player-season of the last 3 complete seasons

# Arena scorers started logging far more giveaways and fewer takeaways in 2024-25 (forwards'
# median giveaways per 60 at 5v5 went from 1.4 to 2.9, takeaways from 1.7 to 0.95), so these
# can't share a pool across seasons. Each player-season is ranked against its own season
# instead; those percentiles then pool like any other. A season still in progress uses the
# latest complete season, which is scored the same way.
SEASON_RELATIVE = frozenset({"takeaways", "giveaways", "take_give_ratio", "giveaway_xga",
                             "oz_takeaways", "dz_takeaways", "oz_giveaways", "dz_giveaways"})


def _ranked_against(stat: str, season, player_season: int | None = None):
    """The pool to rank ``stat`` against: ``season`` itself, or for a season-relative stat in
    a multi-season pool, the player's own season (the latest complete one, if his isn't)."""
    if stat not in SEASON_RELATIVE or len(_seasons(season)) == 1:
        return season
    done = complete_seasons()
    y = player_season if player_season is not None else max(_seasons(season))
    return [y] if y in done else [max([s for s in done if s <= y] or done)]


def _defs() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql("SELECT * FROM stat_defs", con).set_index("stat")


def _strength_for(con, stat: str, strength: str) -> str:
    """Edge stats only exist for all situations; fall back when asked for 5v5."""
    have = [r[0] for r in con.execute("SELECT DISTINCT strength FROM stats WHERE stat = ?", (stat,))]
    if not have:
        raise KeyError(f"unknown stat {stat!r}; run `nhl-microstats stats` to list them")
    return strength if strength in have else have[0]


def complete_seasons() -> list[int]:
    """Seasons with a full schedule in the database (someone played 70+ games)."""
    with connect() as con:
        return [r[0] for r in con.execute(
            "SELECT season FROM player_seasons GROUP BY season HAVING MAX(gp) >= 70 ORDER BY season")]


def _seasons(season) -> list[int]:
    """A season spec -> start years. 'last3' = the three most recent complete
    seasons, 'all' = every complete season; a list passes through."""
    if isinstance(season, (list, tuple)):
        return sorted(start_year(x) for x in season)
    spec = str(season).lower()
    if spec in ("all", "last3"):
        done = complete_seasons()
        return done if spec == "all" else done[-3:]
    return [start_year(season)]


def _position(position: str) -> str:
    pos = str(position).upper()
    if pos not in curves.POSITIONS:
        raise ValueError(f"position must be 'F' (forwards) or 'D' (defensemen), not {position!r}")
    return pos


def pool(stat: str, season=REFERENCE, position: str = "F", strength: str = "5v5",
         per60: bool = True, min_toi: float | None = None) -> tuple[np.ndarray, dict]:
    """The qualified forwards' or defensemen's values for one stat, plus a description of the pool."""
    position = _position(position)
    seasons = _seasons(season)
    defs = _defs()
    with connect() as con:
        strength = _strength_for(con, stat, strength)
        q = f"""SELECT s.value, p.toi_all, p.toi_5v5, p.pos_group FROM stats s
                JOIN player_seasons p USING (player_id, season)
                WHERE s.stat = ? AND s.strength = ? AND s.season IN ({','.join('?' * len(seasons))})"""
        df = pd.read_sql(q, con, params=[stat, strength, *seasons])
    toi = df[f"toi_{strength}"]
    min_toi = MIN_TOI[strength] if min_toi is None else min_toi
    df = df[(toi >= min_toi * 60) & (df["pos_group"] == position)]
    rate = per60 and bool(defs.loc[stat, "per60"])
    values = (df["value"] / df[f"toi_{strength}"] * 3600) if rate else df["value"]
    info = {"stat": stat, "label": defs.loc[stat, "label"], "seasons": seasons, "position": position,
            "strength": strength, "per60": rate, "min_toi": min_toi, "n": len(values),
            "higher_is_better": bool(defs.loc[stat, "higher_is_better"])}
    return np.sort(values.to_numpy(dtype=float)), info


def rank(values: np.ndarray, value: float) -> float:
    """Percentile of ``value`` within sorted ``values`` (ties count half)."""
    if len(values) == 0:
        return float("nan")
    below = np.searchsorted(values, value, side="left")
    at_or_below = np.searchsorted(values, value, side="right")
    return 100 * (below + 0.5 * (at_or_below - below)) / len(values)


def curve_for(values: np.ndarray, info: dict) -> curves.Curve:
    """The stored curve for this pool, or one fitted on the spot for
    non-default pools (totals, a custom TOI cutoff, or a stat added since)."""
    default = info["min_toi"] == MIN_TOI[info["strength"]] and (info["per60"] or not _defs().loc[info["stat"], "per60"])
    if default:
        season = curves.key(info["seasons"])
        with connect() as con:
            c = curves.load(con, info["stat"], info["strength"], season, info["position"])
        if c is not None:
            return c
    return curves.fit(values)


def _oriented(pct: float, info: dict) -> float:
    """Flip a percentile for stats where less is better, so higher is always better."""
    return pct if info["higher_is_better"] else 100 - pct


def percentile(stat: str, value: float, season=REFERENCE, position: str = "F", strength: str = "5v5",
               per60: bool = True, min_toi: float | None = None) -> dict:
    """Where ``value`` would rank among qualified forwards (``position="F"``)
    or defensemen (``"D"``). Returns the fitted
    percentile, the raw mid-rank, and the pool's size, median and 90th
    percentile for context. Season-relative stats (takeaways and giveaways) in a
    multi-season pool rank against its latest season."""
    values, info = pool(stat, _ranked_against(stat, season), position, strength, per60, min_toi)
    if len(values) == 0:
        raise ValueError(f"no qualified players for {stat} in {info['seasons']}; is that season built?")
    curve = curve_for(values, info)
    pct, raw = _oriented(curve(value), info), _oriented(rank(values, value), info)
    return info | {
        "value": value,
        "percentile": round(pct, 1),
        "empirical_percentile": round(raw, 1),
        "curve_mae": round(curve.mae, 2),
        "pool_median": float(np.median(values)) if len(values) else None,
        # the value a player needs for the 90th percentile (the 10th raw percentile when less is better)
        "pool_p90": float(np.percentile(values, 90 if info["higher_is_better"] else 10)) if len(values) else None,
    }


def season_table(season, strength: str = "5v5", per60: bool = True, pool_season=None) -> pd.DataFrame:
    """``player_table`` for every skater in a season at once: one row per (player, stat)
    with value and percentile. Each pool and curve is loaded once, not once per player."""
    y = start_year(season)
    spec0 = REFERENCE if pool_season is None else pool_season
    with connect() as con:
        ps_ = pd.read_sql("SELECT * FROM player_seasons WHERE season = ?", con, params=[y]).set_index("player_id")
        mine = pd.read_sql("SELECT player_id, strength, stat, value FROM stats WHERE season = ?", con, params=[y])
    defs = _defs()
    out = []
    for stat, s in mine.groupby("stat"):
        if stat not in defs.index:
            continue
        st = strength if (s["strength"] == strength).any() else s["strength"].iloc[0]
        s = s[s["strength"] == st].set_index("player_id")["value"]
        for pos in curves.POSITIONS:
            ids = ps_.index[ps_["pos_group"] == pos].intersection(s.index)
            if ids.empty:
                continue
            values, info = pool(stat, _ranked_against(stat, spec0, y), pos, st, per60)
            if len(values) == 0:
                continue
            v = s.loc[ids].astype(float)
            if info["per60"]:
                toi = ps_.loc[ids, f"toi_{st}"].astype(float)
                v = (v / toi * 3600).where(toi > 0)
            ok = np.isfinite(v.to_numpy())
            pct = np.full(len(v), np.nan)
            if ok.any():
                pct[ok] = curve_for(values, info)(v.to_numpy()[ok])
                pct = pct if info["higher_is_better"] else 100 - pct
            out.append(pd.DataFrame({"player_id": ids, "stat": stat, "label": info["label"], "strength": st,
                                     "value": v.to_numpy(), "percentile": pct}))
    t = pd.concat(out, ignore_index=True)
    t.attrs["players"] = ps_
    return t


def player_table(player: str | int, season="2025", strength: str = "5v5", per60: bool = True,
                 pool_season=None) -> pd.DataFrame:
    """Every stat for one player-season, each ranked against their position group.
    ``pool_season`` is the comparison pool (default: the last three complete
    seasons), so a few games of a new season rank against full-season curves."""
    y = start_year(season)
    with connect() as con:
        ps_ = pd.read_sql("SELECT * FROM player_seasons WHERE season = ?", con, params=[y])
    if str(player).isdigit():
        match = ps_[ps_["player_id"] == int(player)]
    else:
        match = ps_[ps_["name"].str.contains(str(player), case=False, regex=False)]
    if match.empty:
        raise LookupError(f"no skater matching {player!r} in {y}-{(y + 1) % 100}")
    if len(match) > 1:
        raise LookupError(f"{len(match)} skaters match {player!r}: {', '.join(match['name'].head(8))}")
    p = match.iloc[0]

    with connect() as con:
        mine = pd.read_sql("SELECT strength, stat, value FROM stats WHERE player_id = ? AND season = ?",
                           con, params=[int(p["player_id"]), y])
    rows = []
    defs = _defs()
    for stat in defs.index:
        have = mine[mine["stat"] == stat]
        if have.empty:
            continue
        row = have[have["strength"] == strength]
        row = row if len(row) else have
        spec = _ranked_against(stat, REFERENCE if pool_season is None else pool_season, y)
        values, info = pool(stat, spec, p["pos_group"], row["strength"].iloc[0], per60)
        v = row["value"].iloc[0]
        if info["per60"]:
            toi = p[f"toi_{info['strength']}"]
            v = v / toi * 3600 if toi else float("nan")
        pct = _oriented(curve_for(values, info)(v), info) if np.isfinite(v) else float("nan")
        rows.append({"category": defs.loc[stat, "category"], "stat": stat, "label": info["label"],
                     "strength": info["strength"], "value": v, "percentile": pct,
                     "higher_is_better": info["higher_is_better"]})
    out = pd.DataFrame(rows)
    out.attrs["player"] = p.to_dict()
    return out


# --------------------------------------------------------------------------
# CLI output
# --------------------------------------------------------------------------

def _ordinal(x: float) -> str:
    n = min(max(int(round(x)), 1), 99)   # a percentile below or above everyone shows as 1st or 99th
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def season_name(seasons) -> str:
    one = lambda y: f"{y}-{(y + 1) % 100:02d}"
    return one(seasons[0]) if len(seasons) == 1 else f"{one(seasons[0])} to {one(seasons[-1])}"


def print_percentile(stat, value, **kwargs) -> None:
    r = percentile(stat, value, **kwargs)
    pos = {"F": "forwards", "D": "defensemen"}[r["position"]]
    unit = " per 60" if r["per60"] else ""
    basis = f"{r['strength']}, season total" if not r["per60"] and _defs().loc[stat, "per60"] else r["strength"]
    print(f"\n{value:g} {r['label'].lower()}{unit} ({basis}) is the "
          f"{_ordinal(r['percentile'])} percentile among {pos}, {season_name(r['seasons'])}.")
    if not r["higher_is_better"]:
        print("Less is better here, so the percentile is flipped: higher means fewer than most of the pool.")
    print(f"Pool: {r['n']} {pos} with {r['min_toi']:g}+ min at {r['strength']}; "
          f"median {r['pool_median']:.2f}, 90th percentile {r['pool_p90']:.2f}.")
    print(f"Logistic curve fit: off the raw rank by {r['curve_mae']:.1f} points on average "
          f"(raw rank for this value: {_ordinal(r['empirical_percentile'])}).\n")


def print_player(name, season="2025", strength="5v5", per60=True, vs=REFERENCE) -> None:
    t = player_table(name, season, strength, per60, pool_season=vs)
    p = t.attrs["player"]
    y = start_year(season)
    print(f"\n{p['name']}  {p['position']}  {p['team']}  {y}-{(y + 1) % 100:02d}  "
          f"GP {p['gp']}  TOI {p['toi_all'] / 60:.0f} min ({p['toi_5v5'] / 60:.0f} at 5v5)")
    print(f"Percentiles vs {'defensemen' if p['pos_group'] == 'D' else 'forwards'}, {season_name(_seasons(vs))}"
          f"{', per 60' if per60 else ', season totals'}. Higher is always better; 'less' marks stats where fewer is better (flipped).\n")
    for cat, g in t.groupby("category", sort=False):
        print(cat.upper())
        for _, r in g.iterrows():
            bar = "#" * int(r["percentile"] // 5) if np.isfinite(r["percentile"]) else ""
            less = "" if r["higher_is_better"] else "less"
            print(f"  {r['label']:<38} {r['value']:>8.2f}  {_ordinal(r['percentile']):>5}  {less:<4}  {bar}")
        print()


def print_stats() -> None:
    from .db import TRACKED
    with connect() as con:
        have = pd.read_sql("""SELECT d.stat, d.label, d.category, GROUP_CONCAT(DISTINCT s.strength) AS strengths
                              FROM stat_defs d JOIN stats s USING (stat) GROUP BY d.stat ORDER BY d.category""", con)
    print(have.to_string(index=False))
    missing = [s for s in TRACKED if s not in set(have["stat"])]
    if missing:
        print("\nNot in the NHL's data (import with `nhl-microstats import-tracked`):\n  " + ", ".join(missing))
