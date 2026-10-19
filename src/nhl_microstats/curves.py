"""Percentile curves fitted by logistic regression.

For each pool (stat x strength x season x position group) the curve is

    percentile(v) = 100 * sigmoid(a + b * z(v) + c * z(v)^3)

where z is a Yeo-Johnson transform of v, standardized on the pool. The
transform takes the skew out of rate stats (most are bunched near zero with a
long right tail). a, b and c come from a logistic regression on "is this player
below threshold t?" over every pool player and 60 thresholds spread across the
pool; ties count half each way, matching the empirical mid-rank. The cubic term
bends the tails for pools one slope can't follow (category averages, bounded
shares); it is only kept when b > 0 and c >= 0, so the curve stays monotone, and
it is capped at |z| = 3 so values far outside the pool extrapolate on the slope alone.

Zero-inflated stats (most defensemen have no zone carries on goals; most players
never fight) put a big share of the pool on one value, a jump no smooth curve
can make. For those the curve is a hurdle: the share p0 at the pool minimum gets
its mid-rank (50 * p0) exactly, and the curve is fitted to the rest and squeezed
into the remaining 100 * (1 - p0) points. Each pool keeps whichever of the plain,
cubic and hurdle fits sits closest to the raw rank.

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
HURDLE_SHARE = 0.02   # try a hurdle when at least this share of the pool sits on its minimum
CUBIC_Z = 3.0         # the cubic term stops growing past |z| = 3, so far outside the pool only the slope extrapolates


def _cube(z):
    return np.clip(z, -CUBIC_Z, CUBIC_Z) ** 3


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
    c REAL DEFAULT 0, p0 REAL DEFAULT 0, floor REAL,
    PRIMARY KEY (stat, strength, season, pos));
"""
COLUMNS = ["lam", "mu", "sd", "a", "b", "n", "mae", "max_err", "c", "p0", "floor"]


def _migrate(con: sqlite3.Connection) -> None:
    """Add the cubic and hurdle columns to a curves table from before they existed."""
    have = {r[1] for r in con.execute("PRAGMA table_info(curves)")}
    for col, decl in (("c", "REAL DEFAULT 0"), ("p0", "REAL DEFAULT 0"), ("floor", "REAL")):
        if have and col not in have:
            con.execute(f"ALTER TABLE curves ADD COLUMN {col} {decl}")


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
    c: float = 0.0                  # cubic term
    p0: float = 0.0                 # hurdle: share of the pool at ``floor`` (0 = no hurdle)
    floor: float | None = None

    def z(self, v):
        y = sps.yeojohnson(np.atleast_1d(np.asarray(v, dtype=float)), self.lam)
        return (y - self.mu) / self.sd

    def __call__(self, v):
        """Percentile (0-100) of one value or an array of values."""
        x = np.atleast_1d(np.asarray(v, dtype=float))
        hurdle = bool(self.p0) and self.floor is not None
        z = self.z(np.maximum(x, self.floor) if hurdle else x)
        p = 100 / (1 + np.exp(-(self.a + self.b * z + (self.c or 0.0) * _cube(z))))
        if hurdle:
            p = np.where(x <= self.floor, 50 * self.p0, 100 * self.p0 + (1 - self.p0) * p)
        return float(p[0]) if np.ndim(v) == 0 else p


def empirical(values: np.ndarray, v) -> np.ndarray:
    """Mid-rank percentile of each v within sorted ``values``."""
    below = np.searchsorted(values, v, side="left")
    upto = np.searchsorted(values, v, side="right")
    return 100 * (below + 0.5 * (upto - below)) / len(values)


def fit(values) -> Curve:
    """The closest of the plain, cubic and (for zero-inflated pools) hurdle curves."""
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) < 2 or np.ptp(x) == 0:
        raise ValueError("need at least two distinct values to fit a curve")
    fits = [_fit(x, x, cubic=False), _fit(x, x, cubic=True)]
    p0 = float(np.mean(x == x[0]))
    rest = x[x > x[0]]
    if p0 >= HURDLE_SHARE and len(rest) >= MIN_POOL // 2 and np.ptp(rest) > 0:
        fits.append(_fit(x, rest, cubic=True, p0=p0, floor=float(x[0])))
    return min((c for c in fits if c is not None), key=lambda c: c.mae)


def _fit(x: np.ndarray, part: np.ndarray, cubic: bool, p0: float = 0.0, floor: float | None = None) -> Curve | None:
    """Fit a curve to ``part`` (all of ``x``, or the values above a hurdle) and score it on ``x``."""
    with np.errstate(all="ignore"):
        lam = float(sps.yeojohnson_normmax(part))
    y = sps.yeojohnson(part, lam)
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
    X = np.column_stack([feats, _cube(feats)]) if cubic else feats.reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X, labels, sample_weight=weights)
    b, c3 = float(lr.coef_[0, 0]), (float(lr.coef_[0, 1]) if cubic else 0.0)
    if b <= 0 or c3 < 0:     # not monotone increasing
        return None

    c = Curve(lam, mu, sd, float(lr.intercept_[0]), b, n=len(x), c=c3, p0=p0, floor=floor)
    err = np.abs(c(x) - empirical(x, x))
    c.mae, c.max_err = float(err.mean()), float(err.max())
    return c


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def load(con: sqlite3.Connection, stat: str, strength: str, season: int, pos: str) -> Curve | None:
    try:
        row = con.execute(f"SELECT {', '.join(COLUMNS)} FROM curves WHERE stat = ? AND strength = ? "
                          "AND season = ? AND pos = ?", (stat, strength, season, pos)).fetchone()
    except sqlite3.OperationalError:     # no curves table yet
        return None
    return Curve(*row) if row else None


def fit_all(stats: list[str] | None = None) -> pd.DataFrame:
    """Fit and store a curve for every pool at the default TOI cutoffs, as rates per
    60: each season on its own, the last three complete seasons pooled (the
    default reference), and every complete season pooled."""
    from .db import DB_PATH, connect
    from .percentile import SEASON_RELATIVE, _seasons, pool

    with connect() as con:
        con.executescript(SCHEMA)
        _migrate(con)
        defs = pd.read_sql("SELECT stat FROM stat_defs", con)["stat"].tolist()
        have = pd.read_sql("SELECT DISTINCT stat, strength FROM stats", con)
        seasons = [r[0] for r in con.execute("SELECT DISTINCT season FROM player_seasons ORDER BY season")]
    have = have[have["stat"].isin(stats or defs)]

    rows = []
    for stat, strength in have.itertuples(index=False):
        # Season-relative stats are only ever ranked against one season, so they get no pooled curve.
        pooled = [] if stat in SEASON_RELATIVE else ["last3", "all"]
        specs = {key(_seasons(spec)): spec for spec in [*seasons, *pooled]}
        for season, spec in specs.items():
            for pos in POSITIONS:
                values, _ = pool(stat, spec, pos, strength, per60=True)
                if len(values) < MIN_POOL or np.ptp(values) == 0:
                    continue
                c = fit(values)
                rows.append((stat, strength, season, pos, *(getattr(c, k) for k in COLUMNS)))
    with sqlite3.connect(DB_PATH) as con:
        if stats:
            con.executemany("DELETE FROM curves WHERE stat = ?", [(s,) for s in stats])
        else:
            con.execute("DELETE FROM curves")
        con.executemany(f"INSERT INTO curves (stat, strength, season, pos, {', '.join(COLUMNS)}) "
                        f"VALUES ({', '.join('?' * (4 + len(COLUMNS)))})", rows)
    out = pd.DataFrame(rows, columns=["stat", "strength", "season", "pos", *COLUMNS])
    log.info("fitted %d curves; mean error %.2f percentile points, worst %.1f",
             len(out), out["mae"].mean() if len(out) else float("nan"), out["max_err"].max() if len(out) else float("nan"))
    return out
