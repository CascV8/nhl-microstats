"""Offline tests on small hand-built games."""

import numpy as np
import pandas as pd
import pytest

from nhl_microstats import build, curves, db, percentile


def _events(rows):
    """Minimal play-by-play: (period_seconds, event_type, x_coord, team)."""
    df = pd.DataFrame(rows, columns=["period_seconds", "event_type", "x_coord", "event_team"])
    n = len(df)
    return df.assign(game_id=1, period=1, sort_order=range(n), event_id=range(n), y_coord=0.0,
                     home_team="HOM", away_team="AWY", home_team_defending_side="left")


def test_rush_and_cycle_clocks():
    # Home attacks toward +x. Takeaway in the neutral zone, shot 3 s later = rush;
    # a shot 15 s after the last out-of-zone event = cycle; one right after it = rebound.
    pbp = _events([
        (0, "faceoff", 0.0, "HOM"),
        (20, "takeaway", 0.0, "HOM"),
        (23, "shot-on-goal", 60.0, "HOM"),
        (38, "missed-shot", 70.0, "HOM"),
        (40, "shot-on-goal", 75.0, "HOM"),
    ])
    shots = pbp[pbp["event_type"].isin(build.ATTEMPTS)].assign(is_home=1)
    flags = build.classify(pbp, shots)
    assert flags["rush"].tolist() == [1, 0, 0]
    assert flags["cycle"].tolist() == [0, 1, 0]


def test_shot_from_outside_blue_line_is_not_its_own_rush():
    pbp = _events([(0, "faceoff", 0.0, "HOM"), (30, "shot-on-goal", 10.0, "HOM")])
    shots = pbp.iloc[[1]].assign(is_home=1)
    assert build.classify(pbp, shots)["rush"].tolist() == [0]


def test_toi_counts_5v5_only_with_five_skaters_and_a_goalie():
    skaters = set(range(1, 13))
    rows = [(p, "HOM", 0, 100) for p in range(1, 6)] + [(p, "AWY", 0, 100) for p in range(6, 11)]
    rows += [(98, "HOM", 0, 100), (99, "AWY", 0, 100)]          # goalies
    rows += [(11, "HOM", 50, 100)]                               # extra skater: 6v5 for 50 s
    shifts = pd.DataFrame(rows, columns=["player_id", "team_abbrev", "start_seconds", "end_seconds"]).assign(game_id=1, period=1)
    t = build.toi(shifts, skaters)
    assert t.loc[1, "toi_all"] == 100
    assert t.loc[1, "toi_5v5"] == 50
    assert t.loc[11, "toi_5v5"] == 0
    assert 98 not in t.index


def test_rank_is_midrank():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile.rank(values, 0) == 0
    assert percentile.rank(values, 5) == 100
    assert percentile.rank(values, 2.5) == 50
    assert percentile.rank(values, 2) == pytest.approx(37.5)   # 1 below + half of 1 tie, of 4


def test_curve_fits_a_skewed_pool():
    rng = np.random.default_rng(0)
    values = rng.gamma(2.0, 1.5, 400)                  # right-skewed, like most rate stats
    c = curves.fit(values)
    assert c.mae < 1.5
    grid = np.linspace(-5, 40, 200)
    assert np.all(np.diff(c(grid)) >= 0)               # monotone
    assert 0 < c(-5) < 1 and 99 < c(40) < 100          # extrapolates without pinning
    assert c(np.median(values)) == pytest.approx(50, abs=2)


def test_percentile_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "m.db")
    db.init()
    n = 100
    ps_ = pd.DataFrame({"player_id": range(n), "season": 2025, "name": [f"P{i}" for i in range(n)],
                        "position": ["C"] * 50 + ["D"] * 50, "team": "X", "gp": 82,
                        "toi_all": 60_000.0, "toi_5v5": 45_000.0})
    stats = pd.concat([pd.DataFrame({"player_id": range(n), "season": 2025, "strength": "5v5",
                                     "stat": stat, "value": np.arange(n, dtype=float)})
                       for stat in ("rush_shots_for", "rush_shots_against")])
    db.write_season(2025, ps_, stats, pd.DataFrame(columns=["player_id", "stat", "value"]))
    curves.fit_all()

    # Rates per 60 by default: 30 shots in 750 min = 2.4/60. Raw rank: 30 of 50 forwards below, 1 tie.
    r = percentile.percentile("rush_shots_for", 2.4, season=2025, position="F")
    assert r["n"] == 50 and r["empirical_percentile"] == 61.0
    # This toy pool is perfectly uniform, which an S-shaped curve only approximates.
    assert r["percentile"] == pytest.approx(61, abs=5)
    totals = percentile.percentile("rush_shots_for", 30, season=2025, position="F", per60=False)
    assert totals["percentile"] == pytest.approx(61, abs=5)

    # Less is better for "against" stats, so the same value gets the mirrored percentile.
    against = percentile.percentile("rush_shots_against", 2.4, season=2025, position="F")
    assert against["percentile"] == pytest.approx(100 - r["percentile"], abs=0.1)
    assert against["empirical_percentile"] == 39.0

    # Forwards and defensemen are separate pools: the same value ranks against each group only.
    d = percentile.percentile("rush_shots_for", 2.4, season=2025, position="D")
    assert d["n"] == 50 and d["empirical_percentile"] == 0.0
    with pytest.raises(ValueError):
        percentile.percentile("rush_shots_for", 2.4, season=2025, position="all")

    t = percentile.player_table("P99", 2025).set_index("stat")
    assert t.loc["rush_shots_for", "value"] == pytest.approx(99 * 3600 / 45_000)
    assert t.loc["rush_shots_for", "percentile"] > 90
    assert t.loc["rush_shots_for", "percentile"] > percentile.player_table("P98", 2025).set_index("stat").loc["rush_shots_for", "percentile"]

    # The website's batch version gives every player the same numbers as player_table.
    batch = percentile.season_table(2025).set_index(["player_id", "stat"])
    for pid in (3, 60, 99):
        one = percentile.player_table(pid, 2025).set_index("stat")
        for stat in one.index:
            assert batch.loc[(pid, stat), "value"] == pytest.approx(one.loc[stat, "value"])
            assert batch.loc[(pid, stat), "percentile"] == pytest.approx(one.loc[stat, "percentile"])


def test_current_season_starts_in_september():
    import datetime as dt
    from nhl_microstats.update import current_season
    assert current_season(dt.date(2026, 10, 18)) == 2026
    assert current_season(dt.date(2027, 4, 2)) == 2026
    assert current_season(dt.date(2027, 9, 1)) == 2027


def test_zone_takeaways_and_giveaways():
    pbp = pd.DataFrame({
        "event_type": ["takeaway", "takeaway", "giveaway", "giveaway", "takeaway", "hit"],
        "zone_code":  ["O", "D", "D", "O", "N", "O"],          # relative to the player's team
        "player_id":  pd.array([1, 1, 2, 1, 1, 2], dtype="Int64"),
    })
    out = build._zone_puck_events(pbp)
    # Totals count every zone, the neutral-zone takeaway too.
    assert out.loc[1].to_dict() == {"takeaways": 3, "giveaways": 1,
                                    "oz_takeaways": 1, "dz_takeaways": 1, "oz_giveaways": 1, "dz_giveaways": 0}
    assert out.loc[2].to_dict() == {"takeaways": 0, "giveaways": 1,
                                    "oz_takeaways": 0, "dz_takeaways": 0, "oz_giveaways": 0, "dz_giveaways": 1}



def test_goal_tracking_possessions():
    from nhl_microstats import goal_tracking as gt
    # Home attacks toward x = 2400 (inches). A carries over the blue line (x = 1500) and past a
    # stationary defender, passes to B, B shoots into the net; then C fishes the puck out.
    rows = []
    def put(i, pid, team, x, y):
        rows.append((i / 10, pid, team, x, y))
    for i in range(40):
        ax = 1200 + 30 * min(i, 20)
        put(i, 1, "HOM", ax, 510)
        put(i, 2, "HOM", 2050, 560)
        put(i, 3, "HOM", 2200 if i < 34 else 2290, 600 if i < 34 else 520)
        put(i, 9, "AWY", 1600, 560)
        put(i, 99, "AWY", 2260, 510)                                # goalie: not a skater
        if i <= 20:
            puck = (ax + 12, 510)
        elif i < 25:
            puck = (1812 + (2040 - 1812) * (i - 20) / 5, 510 + 50 * (i - 20) / 5)
        elif i < 30:
            puck = (2040, 560)
        elif i < 34:
            puck = (2040 + (2290 - 2040) * (i - 29) / 4, 560 - 50 * (i - 29) / 4)
        else:
            puck = (2290, 520)
        put(i, 0, "", *puck)
    goal = pd.DataFrame(rows, columns=["t", "player_id", "team", "x", "y"])
    stats, segs = gt.goal_stats(goal, "HOM", skaters={1, 2, 3, 9})
    assert segs["player_id"].tolist() == [1, 2]                     # C's pickup after the goal is cut
    assert stats.loc[1, "gt_carries"] == 1 and stats.loc[1, "gt_defenders_beaten"] == 1
    assert stats.loc[9, "gt_beaten"] == 1                           # the defender he skated past
    assert stats.loc[1, "gt_possession_time"] == pytest.approx(2.1, abs=0.2)
    assert 3 not in stats.index


def test_goal_tracking_dekes():
    from nhl_microstats import goal_tracking as gt
    # Carrier skates toward +x; the puck goes from 20 in on his left to 20 in on his right and back.
    frames = 12
    carrier = np.column_stack([1500 + 20 * np.arange(frames), np.full(frames, 510.0)])
    side = np.array([20, 20, 20, -20, -20, -20, 20, 20, 20, 20, 20, 20])
    puck = carrier + np.column_stack([np.full(frames, 15.0), side])
    pressured = np.ones(frames, bool)
    assert gt._dekes(puck, carrier, pressured, 1.0) == 2                  # left-to-right, then back
    assert gt._dekes(puck, carrier, ~pressured, 1.0) == 0                 # nobody close: just stickhandling
    small = carrier + np.column_stack([np.full(frames, 15.0), side / 2])  # a 20 in swing is too small
    assert gt._dekes(small, carrier, pressured, 1.0) == 0


def test_blocks_and_hits():
    pbp = pd.DataFrame({
        "event_type": ["blocked-shot", "blocked-shot", "hit", "hit", "hit", "takeaway"],
        "blocking_player_id": pd.array([4, 4, None, None, None, None], dtype="Int64"),
        "hitting_player_id": pd.array([None, None, 4, 5, 5, None], dtype="Int64"),
        "hittee_player_id": pd.array([None, None, 5, 4, 4, None], dtype="Int64"),
    })
    out = build._blocks_hits(pbp)
    assert out.loc[4].to_dict() == {"blocks": 2, "hits": 1, "hits_taken": 2}
    assert out.loc[5].to_dict() == {"blocks": 0, "hits": 2, "hits_taken": 1}


def test_penalties_skip_fights_and_misconducts():
    pbp = pd.DataFrame({
        "event_type": ["penalty"] * 4 + ["hit"],
        "details_type_code": ["MIN", "MAJ", "MAJ", "MIS", None],
        "desc_key": ["tripping", "boarding", "fighting", "misconduct", None],
        "committed_by_player_id": pd.array([4, 4, 4, 4, None], dtype="Int64"),
        "drawn_by_player_id": pd.array([5, 5, 6, None, None], dtype="Int64"),
    })
    out = build._penalties(pbp)
    assert out.loc[4].to_dict() == {"penalties_taken": 2, "penalties_drawn": 0, "fights": 1}
    assert out.loc[5, "penalties_drawn"] == 2 and 6 not in out.index



def test_quality_of_competition():
    # 100 s period: HOM skaters 1-5 + goalie 90, AWY skaters 6-10 + goalie 91. Skater 10 leaves at 80 s,
    # so the last 20 s are 5-on-4 and don't count.
    rows = [(p, "HOM", 0, 100) for p in [1, 2, 3, 4, 5, 90]] + [(p, "AWY", 0, 100) for p in [6, 7, 8, 9, 91]]
    rows.append((10, "AWY", 0, 80))
    shifts = pd.DataFrame(rows, columns=["player_id", "team_abbrev", "start_seconds", "end_seconds"]).assign(game_id=1, period=1)
    skaters = set(range(1, 11))
    # Prior-free shares: 6 is a 70% player, everyone else 50%.
    big = 1e9
    xgf = pd.Series({p: big for p in skaters}) ; xga = pd.Series({p: big for p in skaters})
    xgf[6], xga[6] = 0.7 * big, 0.3 * big
    q = build.competition(shifts, skaters, xgf, xga)
    assert q.loc[1] == pytest.approx(54, abs=0.01)          # five opponents for the same 80 s: (70 + 4 * 50) / 5
    assert q.loc[6] == pytest.approx(50, abs=0.01)
    assert 90 not in q.index                                # goalies aren't rated


def test_quality_of_competition_skips_unrated_opponents():
    # Opponents with no 5v5 xG yet have no xGF%: they're left out, not counted as 50%.
    rows = [(p, "HOM", 0, 100) for p in [1, 2, 3, 4, 5, 90]] + [(p, "AWY", 0, 100) for p in [6, 7, 8, 9, 10, 91]]
    shifts = pd.DataFrame(rows, columns=["player_id", "team_abbrev", "start_seconds", "end_seconds"]).assign(game_id=1, period=1)
    xgf = pd.Series(0.0, index=range(1, 11)); xga = pd.Series(0.0, index=range(1, 11))
    xgf[6], xga[6] = 6.0, 4.0                              # only 6 has any xG: 60%
    q = build.competition(shifts, set(range(1, 11)), xgf, xga)
    assert q.loc[1] == pytest.approx(60)


def test_goalie_ratings():
    """Goalie cards: GSAA/GP, SV% and GA/GP percentiles within the pool, and an overall curved against it."""
    from nhl_microstats import goalies
    rows = [  # player_id, name, team, gp, sa, ga
        (1, "Good", "VAN", 30, 900, 70), (2, "Avg", "EDM", 30, 900, 90), (3, "Bad", "CGY", 30, 900, 110),
        (4, "Rookie", "SEA", 5, 150, 10),   # under the 20-game pool line: rated against the pool, not part of it
    ]
    df = pd.DataFrame([{"player_id": i, "goalie_full_name": n, "team_abbrevs": t, "games_played": gp, "shots_against": sa,
                        "goals_against": ga, "saves": sa - ga, "wins": 0, "losses": 0, "ot_losses": 0, "shutouts": 0}
                       for i, n, t, gp, sa, ga in rows])
    out = {g["name"]: g for g in goalies.goalie_ratings(df, complete=True)}
    assert out["Good"]["overall"] > out["Avg"]["overall"] > out["Bad"]["overall"]
    assert not out["Rookie"]["qualified"] and out["Good"]["qualified"]
    sv = {m["key"]: m for m in out["Good"]["metrics"]}
    assert sv["sv"]["value"] == pytest.approx(830 / 900, abs=1e-4)
    assert sv["ga_gp"]["pct"] > 50   # fewest goals against: a high percentile
    lg_sv = 1 - (70 + 90 + 110 + 10) / (900 * 3 + 150)
    assert sv["gsaa_gp"]["value"] == pytest.approx((900 * (1 - lg_sv) - 70) / 30, abs=1e-4)
    assert goalies.season_meta(df, complete=True)["pool"] == 3
