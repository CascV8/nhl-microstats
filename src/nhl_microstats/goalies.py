"""Goalie ratings for the goalie cards: GSAA/GP, SV% and GA/GP, each as a percentile within the season's goalies,
and an overall that averages the three and is curved against the same pool, so 50 is a median goalie.

    GSAA/GP  goals saved above average per game: shots against x (1 - league SV%) - goals against, per game
    SV%      saves / shots against
    GA/GP    goals against per game (fewer is better, so its percentile is flipped)

The pool is every goalie with MIN_GP games (fewer while a season is under way); goalies below it are still rated
against the pool, as skaters below the minutes are. Every percentile is curved as the skaters' are: a logistic
curve fitted to the pool (curves.fit), so it is smooth and doesn't pin at 0 or 100. A pool too small to fit
(fewer than curves.MIN_POOL goalies) falls back to mid-ranks: ties share the middle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import curves

MIN_GP = {"complete": 20, "early": 3}
METRICS = [("gsaa_gp", "GSAA/GP", False), ("sv", "SV%", False), ("ga_gp", "GA/GP", True)]   # (key, label, lower is better)


def _midrank(pool: list[float], v: float, lower_better: bool = False) -> float:
    below = sum(1 for x in pool if (x > v if lower_better else x < v))
    same = sum(1 for x in pool if x == v)
    return 100 * (below + same / 2) / len(pool)


def _curve(pool: list[float], lower_better: bool = False):
    """value -> percentile within ``pool``: the fitted curve, or mid-ranks when the pool is too small or flat.
    Lower-is-better stats are fitted on their negatives."""
    sign = -1 if lower_better else 1
    if len(pool) >= curves.MIN_POOL and np.ptp(pool) > 0:
        fitted = curves.fit([sign * x for x in pool])
        return lambda v: float(fitted(sign * v))
    return lambda v: _midrank(pool, v, lower_better)


def goalie_ratings(df: pd.DataFrame, complete: bool) -> list[dict]:
    """One dict per goalie: id, name, team, record, the three metrics with their percentiles, and the overall.
    ``df`` is the NHL goalie summary report (puckscrape's nhl.goalie_stats)."""
    df = df[df["games_played"] > 0]
    if df.empty:
        return []
    lg_sv = 1 - df["goals_against"].sum() / df["shots_against"].sum()
    gp = df["games_played"]
    m = pd.DataFrame({"gsaa_gp": (df["shots_against"] * (1 - lg_sv) - df["goals_against"]) / gp,
                      "sv": df["saves"] / df["shots_against"].where(df["shots_against"] > 0),
                      "ga_gp": df["goals_against"] / gp}, index=df.index)
    min_gp = MIN_GP["complete" if complete else "early"]
    in_pool = gp >= min_gp
    if not in_pool.any():
        return []
    pools = {k: m.loc[in_pool, k].dropna().tolist() for k, _, _ in METRICS}
    fitted = {k: _curve(pools[k], lb) for k, _, lb in METRICS}
    pct = {k: m[k].map(lambda v, k=k: None if pd.isna(v) else fitted[k](v)) for k, _, _ in METRICS}
    avg = pd.concat(pct, axis=1).astype(float).mean(axis=1)
    overall = _curve(avg[in_pool].dropna().tolist())
    out = []
    for i, r in df.iterrows():
        a = avg[i]
        out.append({
            "id": int(r["player_id"]), "name": r["goalie_full_name"], "team": r["team_abbrevs"] or "",
            "gp": int(r["games_played"]), "w": int(r["wins"] or 0), "l": int(r["losses"] or 0), "otl": int(r["ot_losses"] or 0),
            "so": int(r["shutouts"] or 0), "qualified": bool(in_pool[i]),
            "overall": None if pd.isna(a) else round(overall(a), 1),
            "metrics": [{"key": k, "label": label, "value": None if pd.isna(m.at[i, k]) else round(float(m.at[i, k]), 4),
                         "pct": None if pct[k][i] is None else round(pct[k][i], 1)} for k, label, _ in METRICS],
        })
    return sorted(out, key=lambda g: (-(g["overall"] if g["overall"] is not None else -1), -g["gp"]))


def season_meta(df: pd.DataFrame, complete: bool) -> dict:
    """League SV% and the pool's size, for the cards' footnote."""
    df = df[df["games_played"] > 0]
    min_gp = MIN_GP["complete" if complete else "early"]
    lg_sv = 1 - df["goals_against"].sum() / df["shots_against"].sum() if len(df) else None
    return {"league_sv": None if lg_sv is None else round(float(lg_sv), 4), "min_gp": min_gp,
            "pool": int((df["games_played"] >= min_gp).sum())}
