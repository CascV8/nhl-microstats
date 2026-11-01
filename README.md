# nhl-microstats

NHL skater microstats and league percentiles: the numbers behind the player cards on
[pattullometrics](https://pattullometrics.pages.dev). The website itself lives in its own repo; this one
is only the data and the maths.

- **collect**: play-by-play, shift charts, bios and NHL Edge tracking from the NHL's API (via
  [puckscrape](https://github.com/CascV8/Puckscrape)), cached in `data/`.
- **build**: per-player season microstats (rush and cycle shots, zone time, on/off-ice goals and xG, quality of
  competition, takeaways, penalties...), every shot scored with the [xG model](https://github.com/CascV8/xG-model),
  into `data/microstats.db`.
- **goal_tracking**: possession stats from the NHL's goal tracking (time with the puck, carries, dekes,
  defenders beaten), 2023-24 on.
- **curves / percentile**: a logistic curve fitted to each stat's pool (position, strength, comparison seasons),
  so any value reads as a percentile.
- **cards**: a card's numbers: every stat's value and percentile, each category's percentile and an overall rating.
- **goalies**: goalie ratings (GSAx/GP, SV%, GA/GP) as percentiles within a season's goalies.

```
uv sync
uv run nhl-microstats collect 2025          # download a season
uv run nhl-microstats build 2025            # microstats into data/microstats.db
uv run nhl-microstats fit                   # percentile curves
uv run nhl-microstats player McDavid        # one player's microstats and percentiles
```

Data goes in `data/` under the folder it's run from (or `NHL_MICROSTATS_DATA`). The xG model's trained file is
found in the xg-model checkout's `models/`, or `XG_MODELS`.

[team-metrics](https://github.com/CascV8/team-metrics) builds team ranks and lineups on top of this package.
