"""Percentile curves fitted by logistic regression.

For each pool (stat x strength x season x position group) the curve is

    percentile(v) = 100 * sigmoid(a + b * z(v))

where z is a Yeo-Johnson transform of v, standardized on the pool. The
transform takes the skew out of rate stats (most are bunched near zero with a
long right tail), so a single logistic slope fits. a and b come from a
logistic regression on "is this player below threshold t?" over every pool
player and 60 thresholds spread across the pool; ties count half each way,
matching the empirical mid-rank.

A fitted curve is smooth, monotone, and gives sensible answers outside the
range a season happened to produce, where a raw rank would pin at 0 or 100.

Curves are fitted separately for forwards and defensemen: the two groups often
form two humps (ixG, rush shots) that no single logistic fits, and a value means
something different for each, so there is no mixed all-skater curve.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression

log = logging.getLogger("nhl_microstats.curves")

N_THRESHOLDS = 60
MIN_POOL = 20


def key(seasons) -> int:
    """Season key for a pool: the start year for one season, first * 10000 + last
    for several (2023-24 to 2025-26 -> 20232025, like NHL season ids)."""
    seasons = sorted(seasons)
    return seasons[0] if len(seasons) == 1 else seasons[0] * 10000 + seasons[-1]

POSITIONS = ("F", "D")

SCHEMA = """
CREATE TABLE IF NOT EXISTS curves (
    stat TEXT, strength TEXT, season INTEGER, pos TEXT,
    lam REAL, mu REAL, sd REAL, a REAL, b REAL, n INTEGER, mae REAL, max_err REAL,
    PRIMARY KEY (stat, strength, season, pos));
"""


@dataclass
class Curve:
    lam: float
    mu: float
    sd: float
    a: float
    b: float
    n: int = 0
    mae: float = float("nan")       # mean |curve - empirical| over the pool, in percentile points
    max_err: float = float("nan")

    def z(self, v):
        y = sps.yeojohnson(np.atleast_1d(np.asarray(v, dtype=float)), self.lam)
        return (y - self.mu) / self.sd

    def __call__(self, v):
        """Percentile (0-100) of one value or an array of values."""
        p = 100 / (1 + np.exp(-(self.a + self.b * self.z(v))))
        return float(p[0]) if np.ndim(v) == 0 else p


def empirical(values: np.ndarray, v) -> np.ndarray:
    """Mid-rank percentile of each v within sorted ``values``."""
    below = np.searchsorted(values, v, side="left")
    upto = np.searchsorted(values, v, side="right")
    return 100 * (below + 0.5 * (upto - below)) / len(values)


def fit(values) -> Curve:
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) < 2 or np.ptp(x) == 0:
        raise ValueError("need at least two distinct values to fit a curve")
    with np.errstate(all="ignore"):
        lam = float(sps.yeojohnson_normmax(x))
    y = sps.yeojohnson(x, lam)
    mu, sd = float(y.mean()), float(y.std())
    z = (y - mu) / sd

    t = np.unique(np.quantile(z, np.linspace(0.005, 0.995, N_THRESHOLDS)))
    Z = np.repeat(t, len(z))
    X = np.tile(z, len(t))
    tie = np.isclose(X, Z)
    # One row per (player, threshold): label 1 if the player is below the threshold.
    # A tied player adds a half-weight row each way.
    feats = np.concatenate([Z, Z[tie]])
    labels = np.concatenate([(X < Z) & ~tie, np.zeros(tie.sum(), bool)]).astype(int)
    labels[np.flatnonzero(tie)] = 1
    weights = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(feats.reshape(-1, 1), labels, sample_weight=weights)

    c = Curve(lam, mu, sd, float(lr.intercept_[0]), float(lr.coef_[0, 0]), n=len(x))
    err = np.abs(c(x) - empirical(x, x))
    c.mae, c.max_err = float(err.mean()), float(err.max())
    return c


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def load(con: sqlite3.Connection, stat: str, strength: str, season: int, pos: str) -> Curve | None:
    try:
        row = con.execute("SELECT lam, mu, sd, a, b, n, mae, max_err FROM curves WHERE stat = ? AND strength = ? "
                          "AND season = ? AND pos = ?", (stat, strength, season, pos)).fetchone()
    except sqlite3.OperationalError:     # no curves table yet
        return None
    return Curve(*row) if row else None


def fit_all(stats: list[str] | None = None) -> pd.DataFrame:
    """Fit and store a curve for every pool at the default TOI cutoffs, as rates per
    60: each season on its own, the last three complete seasons pooled (the
    default reference), and every complete season pooled."""
    from .db import DB_PATH, connect
    from .percentile import _seasons, pool

    with connect() as con:
        con.executescript(SCHEMA)
        defs = pd.read_sql("SELECT stat FROM stat_defs", con)["stat"].tolist()
        have = pd.read_sql("SELECT DISTINCT stat, strength FROM stats", con)
        seasons = [r[0] for r in con.execute("SELECT DISTINCT season FROM player_seasons ORDER BY season")]
    have = have[have["stat"].isin(stats or defs)]

    rows = []
    for stat, strength in have.itertuples(index=False):
        specs = {key(_seasons(spec)): spec for spec in [*seasons, "last3", "all"]}
        for season, spec in specs.items():
            for pos in POSITIONS:
                values, _ = pool(stat, spec, pos, strength, per60=True)
                if len(values) < MIN_POOL or np.ptp(values) == 0:
                    continue
                c = fit(values)
                rows.append((stat, strength, season, pos, c.lam, c.mu, c.sd, c.a, c.b, c.n, c.mae, c.max_err))
    with sqlite3.connect(DB_PATH) as con:
        if stats:
            con.executemany("DELETE FROM curves WHERE stat = ?", [(s,) for s in stats])
        else:
            con.execute("DELETE FROM curves")
        con.executemany("INSERT INTO curves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    out = pd.DataFrame(rows, columns=["stat", "strength", "season", "pos", "lam", "mu", "sd", "a", "b", "n", "mae", "max_err"])
    log.info("fitted %d curves; mean error %.2f percentile points, worst %.1f",
             len(out), out["mae"].mean() if len(out) else float("nan"), out["max_err"].max() if len(out) else float("nan"))
    return out
