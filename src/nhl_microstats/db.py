"""SQLite store: data/microstats.db.

players_seasons  one row per skater-season: name, position, team, GP, TOI
stats            long format: (player_id, season, strength, stat, value)
stat_defs        what each stat means, where it comes from, which way is good
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd
from puckscrape._util import start_year

from .collect import DATA

log = logging.getLogger("nhl_microstats.db")

DB_PATH = DATA / "microstats.db"

# stat -> (label, category, source, per60 makes sense, higher is better)
STATS = {
    # individual
    "goals":              ("Goals", "shooting", "pbp", True, True),
    "ixg":                ("xG (individual)", "shooting", "xg-model", True, True),
    "shot_attempts":      ("Shot attempts (individual)", "shooting", "pbp", True, True),
    "fenwick":            ("Unblocked shot attempts (Fenwick)", "shooting", "pbp", True, True),
    "shots_on_goal":      ("Shots on goal", "shooting", "pbp", True, True),
    "rush_shots":         ("Rush shots (individual)", "shooting", "pbp", True, True),
    "cycle_shots":        ("Cycle shots (individual)", "shooting", "pbp", True, True),
    "shooting_pct":       ("Shooting %", "shooting", "pbp", False, True),
    "primary_assists":    ("Primary assists", "passing", "pbp", True, True),
    "secondary_assists":  ("Secondary assists", "passing", "pbp", True, True),
    "primary_assist_xg":  ("xG of primary-assisted goals", "passing", "xg-model", True, True),
    # puck battles: takeaways/giveaways in all zones, and by zone relative to the player's team
    "takeaways":          ("Takeaways", "puck battles", "pbp", True, True),
    "giveaways":          ("Giveaways", "puck battles", "pbp", True, False),
    "giveaway_xga":       ("xG against off his giveaways", "puck battles", "pbp+xg-model", True, False),
    "take_give_ratio":    ("Takeaway : giveaway ratio", "puck battles", "pbp", False, True),
    "oz_takeaways":       ("Offensive-zone takeaways", "puck battles", "pbp", True, True),
    "dz_takeaways":       ("Defensive-zone takeaways", "puck battles", "pbp", True, True),
    "oz_giveaways":       ("Offensive-zone giveaways", "puck battles", "pbp", True, False),
    "dz_giveaways":       ("Defensive-zone giveaways", "puck battles", "pbp", True, False),
    # defence
    "blocks":             ("Shot blocks", "defence", "pbp", True, True),
    "block_share":        ("Block share (of on-ice shots against)", "defence", "pbp+shifts", False, True),
    "hits":               ("Hits", "defence", "pbp", True, True),
    "hits_taken":         ("Hits taken", "physical", "pbp", True, True),
    "fights":             ("Fights", "physical", "pbp", True, True),
    "penalties_taken":    ("Penalties taken", "discipline", "pbp", True, False),
    "penalties_drawn":    ("Penalties drawn", "discipline", "pbp", True, True),
    # on ice
    "shots_for":          ("On-ice shot attempts for", "on-ice", "pbp+shifts", True, True),
    "shots_against":      ("On-ice shot attempts against", "on-ice", "pbp+shifts", True, False),
    "fenwick_for":        ("On-ice unblocked attempts for", "on-ice", "pbp+shifts", True, True),
    "fenwick_against":    ("On-ice unblocked attempts against", "on-ice", "pbp+shifts", True, False),
    # on-ice shares (already a %)
    "fenwick_pct":        ("Fenwick % (on-ice)", "on-ice", "pbp+shifts", False, True),
    "xgf_pct":            ("xGF% (on-ice)", "on-ice", "xg-model+shifts", False, True),
    "xgf":                ("On-ice xG for", "on-ice", "xg-model+shifts", True, True),
    "xga":                ("On-ice xG against", "on-ice", "xg-model+shifts", True, False),
    "hd_xga":             ("On-ice high-danger xG against", "on-ice", "xg-model+shifts", True, False),
    "goals_for":          ("On-ice goals for", "on-ice", "pbp+shifts", True, True),
    "goals_against":      ("On-ice goals against", "on-ice", "pbp+shifts", True, False),
    "rush_shots_for":     ("On-ice rush shots for", "on-ice", "pbp+shifts", True, True),
    "rush_shots_against": ("On-ice rush shots against", "on-ice", "pbp+shifts", True, False),
    "cycle_shots_for":    ("On-ice cycle shots for", "on-ice", "pbp+shifts", True, True),
    "cycle_shots_against": ("On-ice cycle shots against", "on-ice", "pbp+shifts", True, False),
    # on-ice GA/60 as a % of the team's GA/60 with him off the ice, same games (already a ratio)
    "ga_on_off_pct":      ("Goals against on/off ice %", "on-ice", "pbp+shifts", False, False),
    # on-ice unblocked attempts against per 60 as a % of the team's with him off the ice, same games (5v5 only)
    "fa_on_off_pct":      ("Shots against on/off ice % (Fenwick)", "on-ice", "pbp+shifts", False, False),
    # on-ice xGF% minus the team's xGF% with him off the ice, same games (percentage points; 5v5 only)
    "xgf_pct_rel":        ("xGF% on/off ice (rel)", "on-ice", "xg-model+shifts", False, True),
    # quality of competition: opponents' 5v5 xGF%, weighted by shared 5v5 ice time (already a %)
    "qoc_xgf_pct":        ("Quality of competition (opponents' xGF%)", "competition", "xg-model+shifts", False, True),
    # NHL Edge (all situations, season values, not counts)
    "max_skating_speed":  ("Max skating speed (mph)", "edge", "nhl-edge", False, True),
    "bursts_over_20mph":  ("Speed bursts 20+ mph", "edge", "nhl-edge", True, True),
    "shot_strength":      ("Shot strength (mph)", "edge", "nhl-edge", False, True),
    "distance_skated":    ("Distance skated (mi)", "edge", "nhl-edge", True, True),
    "max_game_distance":  ("Most distance in a game (mi)", "edge", "nhl-edge", False, True),
    "oz_time_pct":        ("Offensive-zone time %", "edge", "nhl-edge", False, True),
    "oz_time_pct_ev":     ("Offensive-zone time % (even strength)", "edge", "nhl-edge", False, True),
    "nz_time_pct":        ("Neutral-zone time %", "edge", "nhl-edge", False, True),
    "dz_time_pct":        ("Defensive-zone time %", "edge", "nhl-edge", False, False),
}

# Stats the NHL doesn't publish -- they come from video tracking (e.g. All Three Zones).
# Load them with `nhl-microstats import-tracked file.csv` using these column names.
TRACKED = {
    "expected_assists":     ("Expected assists (xG of passes leading to shots)", True),
    "shot_assists":         ("Shot assists", True),
    "rush_shot_assists":    ("Rush shot assists", True),
    "high_danger_passes":   ("High-danger passes", True),
    "passes":               ("Passes", True),
    "zone_entries":         ("Zone entries", True),
    "zone_carry_entries":   ("Zone entries by carry", True),
    "zone_pass_entries":    ("Zone entries by pass", True),
    "zone_exits":           ("Zone exits", True),
    "zone_pass_exits":      ("Zone exits by pass", True),
    "zone_carry_exits":     ("Zone exits by carry", True),
    "entry_denials":        ("Entry denials", True),
    "dz_retrievals":        ("Defensive-zone retrievals", True),
    "oz_retrievals":        ("Offensive-zone retrievals", True),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_seasons (
    player_id INTEGER, season INTEGER, name TEXT, position TEXT, pos_group TEXT, team TEXT,
    gp INTEGER, toi_all REAL, toi_5v5 REAL, PRIMARY KEY (player_id, season));
CREATE TABLE IF NOT EXISTS stats (
    player_id INTEGER, season INTEGER, strength TEXT, stat TEXT, value REAL,
    PRIMARY KEY (player_id, season, strength, stat));
CREATE INDEX IF NOT EXISTS stats_pool ON stats (stat, strength, season);
CREATE TABLE IF NOT EXISTS stat_defs (
    stat TEXT PRIMARY KEY, label TEXT, category TEXT, source TEXT, per60 INTEGER, higher_is_better INTEGER);
"""


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found; run `nhl-microstats build` first")
    return sqlite3.connect(DB_PATH)


def init() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(SCHEMA)
        con.execute("DELETE FROM stat_defs WHERE source != 'import'")
        con.executemany("INSERT OR REPLACE INTO stat_defs VALUES (?, ?, ?, ?, ?, ?)",
                        [(k, *v) for k, v in STATS.items()])
        # Goal-tracking stats are built separately (`nhl-microstats goal-tracking`) and survive rebuilds.
        from .goal_tracking import STATS as GT
        con.executemany("INSERT OR REPLACE INTO stat_defs VALUES (?, ?, 'goal tracking', 'nhl-goal-tracking', ?, ?)",
                        [(k, label, per60, better) for k, (label, per60, better) in GT.items()])
        # Stats dropped from STATS (e.g. top_shot_speed) shouldn't linger in pools.
        con.execute("DELETE FROM stats WHERE stat NOT IN (SELECT stat FROM stat_defs)")


def write_season(season: int, player_seasons: pd.DataFrame, stats: pd.DataFrame, edge: pd.DataFrame) -> None:
    ps_ = player_seasons.assign(pos_group=player_seasons["position"].map(lambda p: "D" if p == "D" else "F"))
    ps_ = ps_[["player_id", "season", "name", "position", "pos_group", "team", "gp", "toi_all", "toi_5v5"]]
    edge = edge[edge["player_id"].isin(ps_["player_id"])].assign(season=season, strength="all")
    rows = pd.concat([stats, edge], ignore_index=True)[["player_id", "season", "strength", "stat", "value"]]
    with sqlite3.connect(DB_PATH) as con:
        # Rebuilds replace the computed stats but keep anything imported by hand.
        computed = tuple(STATS)
        con.execute("DELETE FROM player_seasons WHERE season = ?", (season,))
        con.execute(f"DELETE FROM stats WHERE season = ? AND stat IN ({','.join('?' * len(computed))})", (season, *computed))
        con.executemany("INSERT INTO player_seasons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ps_.itertuples(index=False))
        con.executemany("INSERT INTO stats VALUES (?, ?, ?, ?, ?)", rows.itertuples(index=False))


def import_tracked(csv_path: str, strength: str = "5v5") -> None:
    """Load hand-tracked microstats. The CSV needs a ``season`` column (any
    spelling: 2024, 2024-25, 20242025), a ``player_id`` or ``name`` column,
    and one column per stat. Stats named in TRACKED get proper labels; any
    other numeric column is imported as-is.
    """
    df = pd.read_csv(csv_path)
    df["season"] = df["season"].map(start_year)
    with connect() as con:
        known = pd.read_sql("SELECT player_id, season, name FROM player_seasons", con)
        if "player_id" not in df:
            if "name" not in df:
                raise ValueError("CSV needs a player_id or name column")
            df = df.merge(known, on=["name", "season"], how="left")
            unmatched = df["player_id"].isna()
            if unmatched.any():
                log.warning("%d rows didn't match a player by name: %s", unmatched.sum(),
                            df.loc[unmatched, "name"].head(10).tolist())
                df = df[~unmatched]
        cols = [c for c in df.columns if c not in ("player_id", "season", "name", "team", "position")
                and pd.api.types.is_numeric_dtype(df[c])]
        long = df.melt(["player_id", "season"], cols, var_name="stat").dropna(subset=["value"])
        long = long.astype({"player_id": "int64"}).assign(strength=strength)
        con.executemany("INSERT OR REPLACE INTO stats VALUES (?, ?, ?, ?, ?)",
                        long[["player_id", "season", "strength", "stat", "value"]].itertuples(index=False))
        con.executemany("INSERT OR IGNORE INTO stat_defs VALUES (?, ?, 'tracked', 'import', ?, 1)",
                        [(c, TRACKED.get(c, (c,))[0], int(TRACKED.get(c, (c, True))[1])) for c in cols])
    print(f"Imported {len(long)} values for {len(cols)} stats: {', '.join(cols)}")
    from .curves import fit_all
    fit_all(cols)
