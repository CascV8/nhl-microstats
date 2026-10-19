# nhl-microstats

Skater microstats for the last five NHL regular seasons (2021-22 to 2025-26), stored in SQLite,
with a percentile engine: give it a value, get back where that value would rank in the league.

Built on [puckscrape](../puckscrape) (NHL play-by-play, shift charts, NHL Edge) and
[xg-model](../xg-model) (every unblocked shot is scored with your xG model).

```
uv run nhl-microstats collect        # download pbp + shifts + bios + Edge (~30 min, resumable)
uv run nhl-microstats build          # compute stats -> data/microstats.db (~2 min/season)

uv run nhl-microstats pct rush_shots_for 3.5 --pos F       # 3.5 per 60 at 5v5 among forwards, last 3 seasons
uv run nhl-microstats pct xga 2.4 --pos D --season 2024    # among defensemen, vs 2024-25 only
uv run nhl-microstats pct goals 40 --totals --strength all  # season totals; no --pos ranks vs F and D separately
uv run nhl-microstats pct shooting_pct 12.5 --pos F        # shooting % is used as it is
uv run nhl-microstats player "Cale Makar"                  # 2025-26 numbers vs the last 3 seasons
uv run nhl-microstats cards Buium --season 2026            # early-season card vs the last 3 seasons
uv run nhl-microstats fit            # refit the percentile curves (build does this too)
uv run nhl-microstats stats          # every stat in the database
```

```python
from nhl_microstats.percentile import percentile, player_table

percentile("primary_assists", 1.2, season=2025, position="F", strength="all")["percentile"]
player_table("McDavid", season=2025)
```

## The percentile

Forwards and defensemen are ranked separately: each has its own pool and its own fitted curve,
and there is no mixed all-skater percentile. Without `--pos`, `pct` prints the forward and the
defense percentile one after the other.

The pool for a stat is every skater in the chosen season and position group (F or D) with
enough ice time at that strength (200+ min at 5v5, 300+ min all situations; `--min-toi` changes it).
Values are rates per 60 minutes (`--totals` switches to season totals). Shooting %, Edge speeds, shot strength
and zone-time shares aren't counts, so they're used as they are.

The default comparison pool is every qualified player-season from the **last three complete seasons**
(2023-24 to 2025-26): about 1,375 forwards and 758 defensemen at 5v5, three times a single season.
`--season 2024` (or `--vs 2024` for `player`, `--vs-season 2024` for `cards`) compares against one season,
and `all` pools every complete season. Partial seasons (the current one) never join a pool.
A pooled pool mixes eras: rush rates in particular ran higher in 2023-24 and 2024-25 than in 2025-26,
so a 2025-26 rush number ranks lower against the three-season pool than against its own season.

Percentiles come from a curve fitted to each pool rather than a raw rank:
`percentile = 100 * sigmoid(a + b * z(value))`, where `z` is a Yeo-Johnson transform
(standardized on the pool) that takes the skew out of rate stats, and `a`, `b` come from a logistic
regression on "is this player below threshold t?" across the pool. Curves are fitted for every
stat, strength, position group and pool (each season, the last three, and all five: 798 in all)
and stored in the `curves` table, keyed by start year or `first * 10000 + last` for pooled seasons;
`fit` refits them. Keeping the groups apart also suits the fit: many stats (ixG, rush shots) have separate
forward and defense humps that one logistic can't fit.
On average a single-season curve sits 1.1 percentile points from the raw rank, and a three-season curve 0.85 (worst point 3.7). The largest single-season gaps (up to ~7 points) are
mid-pool in defense pools of ~240 players, where the raw rank itself has a sampling error of about 3 points.
The fitted curve also gives sensible answers beyond the best or worst value in the pool, where a raw rank pins at 0 or 100.
Every result also reports the raw rank (`empirical_percentile`).

For stats where less is better (giveaways; shots, xG, goals, rush and cycle shots against; DZ time) the
percentile is flipped, so a higher percentile always means better: fewer giveaways ranks higher.

## Stats

| Stat | Source |
|---|---|
| `goals`, `shot_attempts`, `shots_on_goal`, `primary_assists`, `secondary_assists` | play-by-play |
| `shooting_pct` | play-by-play: goals / shots on goal x 100 (players without a shot on goal have none) |
| `ixg` | xg-model, individual unblocked shots (empty-net shots aren't scored) |
| `primary_assist_xg` | xg-model: xG of the goals the player primary-assisted |
| `rush_shots`, `cycle_shots` | individual shot attempts, classified as below |
| `oz_takeaways`, `dz_takeaways`, `oz_giveaways`, `dz_giveaways` | play-by-play: takeaways and giveaways in the player's offensive or defensive zone (neutral zone left out) |
| `shots_for/against`, `xgf/xga`, `goals_for/against` | on ice, from play-by-play matched to shift charts |
| `rush_shots_for/against`, `cycle_shots_for/against` | on ice |
| `max_skating_speed`, `bursts_over_20mph`, `shot_strength` (average shot speed), `distance_skated`, `max_game_distance`, `oz/nz/dz_time_pct`, `oz_time_pct_ev` | NHL Edge (all situations) |

Every play-by-play stat exists at `5v5` and `all` strengths.
Goals, assists and TOI match the NHL's official totals.

**Zone takeaways and giveaways:** the NHL logs no puck retrievals or loose-puck recoveries, so
these are the closest free stand-in. They aren't the same thing: a takeaway is stripping an opponent, not
winning a loose puck. Arena scorers also log takeaways and giveaways very unevenly (some rinks record
several times more than others), so read these percentiles more loosely than shot-based ones.

**Rush:** the puck was recorded outside the shooting team's attacking zone in the previous
8 seconds, with no faceoff since. The NHL logs no zone entries, so this only catches rushes where
something (a takeaway, hit, giveaway, shot...) was logged in the neutral or defensive zone just
before. It flags ~6% of shot attempts, well below what video tracking finds (20-30%), so counts are
small and percentiles for rush stats are noisier than the rest. The window is 8 s, not tighter,
because the delay between play and logged event times shifts between seasons (the high-scoring
post-exit shots come 0-6 s after the exit in 2021-22 but 4-12 s in 2024-25).

**Cycle:** 10+ seconds since the puck was last recorded outside the zone or since a faceoff,
and not a rebound or rush. About three quarters of attempts qualify. Play-by-play can't tell a true
cycle from a forecheck recovery, so read this as "sustained in-zone offence".

## Hand-tracked microstats

The NHL publishes no passing, zone-entry, zone-exit, entry-denial or retrieval data, and
NHL Edge doesn't either (it only covers speed, shot speed, distance and zone time).
These need video tracking such as All Three Zones:

`expected_assists`, `shot_assists`, `rush_shot_assists`, `high_danger_passes`, `passes`,
`zone_entries`, `zone_carry_entries`, `zone_pass_entries`, `zone_exits`, `zone_pass_exits`,
`zone_carry_exits`, `entry_denials`, `dz_retrievals`, `oz_retrievals`

If you have that data, load it and the percentile engine handles it like every other stat:

```
uv run nhl-microstats import-tracked tracked.csv     # columns: season, player_id or name, then one per stat
uv run nhl-microstats pct zone_entries 150 --pos F
```

Rebuilding the play-by-play stats keeps imported stats.

## Goal tracking

For every goal the NHL publishes puck and player tracking for the ~14 s before it (10 frames a
second; the data behind nhl.com's goal replays). It only covers goals, but it's real tracking, so
it gives possession stats on the build-up to a team's goals:

| Stat | Meaning |
|---|---|
| `gt_possession_time` | seconds with the puck |
| `gt_carries` | carrying the puck over the offensive blue line |
| `gt_defenders_beaten` | an opposing skater within 10 ft who was ahead of the carrier when he got the puck and behind him when he gave it up |
| `gt_dekes` | the puck swings 2.5+ ft across the carrier's body, side to side, within 0.5 s, with an opponent or the goalie within 10 ft |

Possession is the nearest skater within 5 ft of the puck, with the puck moving at most 40 ft/s
relative to him, for at least 0.2 s. Clips are cut when the puck enters the net. As a check, the
last possession is the scorer, and the teammate before him the primary assister, on most goals
(the build log gives the rates; tips and deflections are too quick to register as possession).
Passes aren't counted here: primary assists and primary-assist xG already cover them.

```
uv run nhl-microstats goal-tracking 2025                         # download (~100 MB a season) and build
uv run nhl-microstats cards "Matthew Schaefer" --goal-tracking   # cards show them only with this flag
```

## Tests

```
uv run pytest
```
