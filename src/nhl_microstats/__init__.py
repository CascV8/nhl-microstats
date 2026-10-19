"""NHL skater microstats and percentiles, built on puckscrape and xg-model."""

import argparse
import logging

SEASONS = ["2021", "2022", "2023", "2024", "2025"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="nhl-microstats", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="download play-by-play, shifts, bios and Edge data")
    c.add_argument("seasons", nargs="*", default=SEASONS, help="start years (default: last 5 seasons)")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--no-edge", action="store_true")

    b = sub.add_parser("build", help="compute microstats and write data/microstats.db")
    b.add_argument("seasons", nargs="*", default=SEASONS)

    p = sub.add_parser("pct", help="percentile of a value, e.g. `pct rush_shots_for 25`")
    p.add_argument("stat")
    p.add_argument("value", type=float)
    p.add_argument("--season", default="last3",
                   help="pool: 'last3' (last 3 complete seasons, default), 'all', or one season's start year")
    p.add_argument("--pos", default=None, choices=["F", "D"], help="forwards or defensemen (default: both)")
    p.add_argument("--strength", default="5v5", choices=["5v5", "all"])
    p.add_argument("--totals", action="store_true", help="value is a season total, not a rate per 60")
    p.add_argument("--min-toi", type=float, default=None, help="minutes to qualify for the pool")

    pl = sub.add_parser("player", help="every microstat and percentile for one player")
    pl.add_argument("name", help="name (or part of it) or player id")
    pl.add_argument("--season", default="2025", help="the player's season")
    pl.add_argument("--vs", default="last3", help="comparison pool: 'last3' (default), 'all', or a start year")
    pl.add_argument("--strength", default="5v5", choices=["5v5", "all"])
    pl.add_argument("--totals", action="store_true", help="season totals instead of rates per 60")

    sub.add_parser("stats", help="list the stats in the database")
    sub.add_parser("fit", help="refit the logistic percentile curves")

    cd = sub.add_parser("cards", help="HTML player cards (default: top 12 F + top 4 D by points)")
    cd.add_argument("players", nargs="*", help="names (or parts) or player ids")
    cd.add_argument("--season", default="2025")
    cd.add_argument("--out", default="cards.html")
    cd.add_argument("--vs-season", default="last3", help="comparison pool: 'last3' (default), 'all', or a start year")
    cd.add_argument("--goal-tracking", action="store_true",
                    help="add the goal-tracking section (run `goal-tracking` for the season first)")

    gtp = sub.add_parser("goal-tracking", help="download the NHL's goal tracking and build possession stats from it")
    gtp.add_argument("seasons", nargs="*", default=["2025"])
    gtp.add_argument("--workers", type=int, default=6)

    i = sub.add_parser("import-tracked", help="add hand-tracked microstats (entries, exits, passes...) from a CSV")
    i.add_argument("csv")
    i.add_argument("--strength", default="5v5", choices=["5v5", "all"])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("puckscrape").setLevel(logging.WARNING)

    if args.cmd == "collect":
        from . import collect
        for s in args.seasons:
            collect.collect_games(s, workers=args.workers)
            collect.collect_bios(s)
        if not args.no_edge:
            import pandas as pd
            for s in args.seasons:
                bios = pd.read_parquet(collect.collect_bios(s))
                collect.collect_edge(s, bios["player_id"])
                collect.collect_shot_speed(s, bios["player_id"])
    elif args.cmd == "build":
        from .build import build
        build(args.seasons)
    elif args.cmd == "pct":
        from .percentile import print_percentile
        for pos in [args.pos] if args.pos else ["F", "D"]:
            print_percentile(args.stat, args.value, season=args.season, position=pos,
                             strength=args.strength, per60=not args.totals, min_toi=args.min_toi)
    elif args.cmd == "player":
        from .percentile import print_player
        print_player(args.name, season=args.season, strength=args.strength, per60=not args.totals, vs=args.vs)
    elif args.cmd == "fit":
        from .curves import fit_all
        fit_all()
    elif args.cmd == "stats":
        from .percentile import print_stats
        print_stats()
    elif args.cmd == "cards":
        from .cards import write_cards
        write_cards(args.players, season=args.season, out=args.out, vs_season=args.vs_season,
                    goal_tracking=args.goal_tracking)
    elif args.cmd == "goal-tracking":
        from .goal_tracking import build_goal_tracking, collect_goal_tracking
        for s in args.seasons:
            collect_goal_tracking(s, workers=args.workers)
            build_goal_tracking(s)
    elif args.cmd == "import-tracked":
        from .db import import_tracked
        import_tracked(args.csv, strength=args.strength)
