"""Team pages: every NHL team's metrics for the season so far, and its current lineup from Daily Faceoff.

    nhl-microstats lineups      # refresh the Daily Faceoff lineups and NHL rosters (at most every REFRESH_HOURS)

The site build (site.py) then writes, for the season under way:

    data/teams/index.json       the 32 teams: abbrev, name, logo, record, overall
    data/teams/<ABBR>.json      one team's page (team.html?team=ABBR)
    logos/<ABBR>.svg            the team's logo, served with the site so the page can read and crop it (the NHL's
                                asset server allows no cross-origin reads)

Team metrics, all from the season so far, not tonight's lineup, each shown as a rank among the 32 teams (1st best):
    P%, PP%, PK%     NHL stats
    categories       each skater category's 5v5 percentile, averaged over every skater who played for the team,
                     weighted by ice time (Competition left out)
    Goaltending      the team's goalies' overall ratings, weighted by games
    overall          the average of a team's metric ranks, ranked again

Lineups: Daily Faceoff's line combinations (the __NEXT_DATA__ JSON on each team's page), matched to NHL player ids
by name against the team's NHL roster. Daily Faceoff's slots: f1-f4 forward lines, d1-d3 pairs, g, pp1-pp2,
pk1-pk2, ir. Anyone else on the roster is an extra.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import requests

from . import collect
from .percentile import season_name

log = logging.getLogger("nhl_microstats.teams")

LINEUPS = collect.DATA / "lineups"          # <ABBR>.json: Daily Faceoff's combinations and the NHL roster, as fetched
LOGOS = collect.DATA / "logos"              # <ABBR>.svg: the NHL's dark-background logos, refreshed every LOGO_DAYS
LOGO_DAYS = 30
REFRESH_HOURS = 6
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
NHL = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"
TEAM_STATS = [("pointPct", "P%"), ("powerPlayPct", "PP%"), ("penaltyKillPct", "PK%")]


def _get(url: str, tries: int = 4, **kw):
    """GET, waiting and retrying when rate-limited (the NHL's API answers 429 to a burst of 32 roster requests)."""
    for attempt in range(tries):
        r = requests.get(url, timeout=30, **kw)
        if r.status_code != 429 or attempt == tries - 1:
            break
        time.sleep(float(r.headers.get("Retry-After") or 0) or 2 ** (attempt + 1))
    r.raise_for_status()
    return r


def standings() -> list[dict]:
    return _get(f"{NHL}/standings/now").json()["standings"]


def slug(name: str) -> str:
    """Daily Faceoff's team slug: the NHL's full name, ASCII, hyphenated ("Montréal Canadiens" -> montreal-canadiens)."""
    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_).strip("-")


def _combinations(team_name: str) -> dict:
    html = _get(f"https://www.dailyfaceoff.com/teams/{slug(team_name)}/line-combinations", headers=UA).text
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise ValueError("no __NEXT_DATA__ on the page")
    return json.loads(m.group(1))["props"]["pageProps"]["combinations"]


def _roster(abbr: str) -> list[list]:
    """[name, id, position code] for everyone on the team's current NHL roster."""
    d = _get(f"{NHL}/roster/{abbr}/current").json()
    return [[f'{p["firstName"]["default"]} {p["lastName"]["default"]}', p["id"], p["positionCode"]]
            for grp in ("forwards", "defensemen", "goalies") for p in d.get(grp, [])]


def refresh_lineups(force: bool = False) -> tuple[str, int]:
    """Fetch each team's lineup and roster when ours is older than REFRESH_HOURS (or always, with ``force``).
    A team that can't be fetched keeps what it had. Returns a fingerprint of every lineup and roster, which changes
    only when one of them does (not when they are merely re-fetched), and how many teams were fetched."""
    LINEUPS.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    fetched_now = 0
    for t in standings():
        abbr, name = t["teamAbbrev"]["default"], t["teamName"]["default"]
        path = LINEUPS / f"{abbr}.json"
        if path.exists() and not force:
            fetched = dt.datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["fetched"])
            if now - fetched < dt.timedelta(hours=REFRESH_HOURS):
                continue
        try:
            entry = {"fetched": now.isoformat(timespec="seconds"), "name": name,
                     "combinations": _combinations(name), "roster": _roster(abbr)}
        except Exception as e:   # one team's page failing shouldn't stop the others
            log.warning("%s: lineup not refreshed (%s)", abbr, e)
            continue
        path.write_text(json.dumps(entry), encoding="utf-8")
        fetched_now += 1
        time.sleep(0.5)   # spaced out, so the NHL's API doesn't rate-limit the rosters
    return lineups_fingerprint(), fetched_now


def lineups_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(LINEUPS.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        h.update(path.stem.encode() + json.dumps([d.get("combinations"), d.get("roster")], sort_keys=True).encode())
    return h.hexdigest()


def _summary(y: int) -> list[dict]:
    """The NHL's team summary for season ``y``, regular season."""
    return _get(f"{STATS}/team/summary", params={"cayenneExp": f"seasonId={y}{y + 1} and gameTypeId=2", "limit": -1}).json()["data"]



def _logo(abbr: str, out: Path) -> None:
    """Copy the team's logo into the site, fetching it when ours is missing or older than LOGO_DAYS."""
    LOGOS.mkdir(parents=True, exist_ok=True)
    path = LOGOS / f"{abbr}.svg"
    age = dt.datetime.now().timestamp() - path.stat().st_mtime if path.exists() else None
    if age is None or age > LOGO_DAYS * 86400:
        try:
            path.write_bytes(_get(f"https://assets.nhle.com/logos/nhl/svg/{abbr}_dark.svg").content)
        except Exception as e:
            log.warning("%s: logo not fetched (%s)", abbr, e)
    if path.exists():
        (out / "logos").mkdir(exist_ok=True)
        (out / "logos" / path.name).write_bytes(path.read_bytes())


# ── Matching Daily Faceoff's names to NHL ids ────────────────────────────────────────────────────────────

def _key(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(".", "").strip()


def _group(pos: str) -> str:
    pos = pos.upper()
    return "G" if pos.startswith("G") else "D" if pos in ("D", "LD", "RD") else "F"


def _matcher(roster: list[list]):
    """name, Daily Faceoff slot -> NHL id, or None. Matches the full name within the position group, then the last
    name (Daily Faceoff writes "Elias Nils Pettersson" for the defenceman). Special-teams and IR slots (sk1, ir1...)
    aren't positions: those match in any group, on the full name only when two players share it."""
    people = [(_key(n), pid, _group(pos)) for n, pid, pos in roster]

    def match(name: str, slot: str):
        k, g = _key(name), (None if slot.lower()[:2] in ("sk", "ir") else _group(slot))
        ok = lambda rg: g is None or rg == g
        full = [pid for n, pid, rg in people if n == k and ok(rg)]
        last = [pid for n, pid, rg in people if n.split()[-1] == k.split()[-1] and ok(rg)]
        if len(full) == 1:
            return full[0]
        if g is None and len(full) > 1:
            return None
        return last[0] if len(last) == 1 else None
    return match


# ── The pages' data ──────────────────────────────────────────────────────────────────────────────────────

def _teams_of(team: str) -> list[str]:
    return [t for t in re.split(r"[/,]", team or "") if t]


def _ranks(values: dict[str, float | None], lower_better: bool = False) -> dict[str, int | None]:
    """Each team's rank, 1 the best; ties share the higher rank; a team without a value has none."""
    have = [v for v in values.values() if v is not None]
    better = (lambda x, v: x < v) if lower_better else (lambda x, v: x > v)
    return {a: None if v is None else 1 + sum(1 for x in have if better(x, v)) for a, v in values.items()}


def export_teams(season_dir: Path, y: int, out: Path) -> int:
    """Write data/teams/ into ``out`` from season ``y``'s exported skaters and goalies (``season_dir``). Only for the
    season under way (Daily Faceoff and the standings are today's). Returns the number of team pages written."""
    table = standings()
    if not table or int(str(table[0]["seasonId"])[:4]) != y:
        log.info("teams: %s isn't the season under way; no team pages", y)
        return 0
    rows = json.loads((season_dir / "index.json").read_text(encoding="utf-8"))["rows"]
    gpath = season_dir / "goalies.json"
    gdata = json.loads(gpath.read_text(encoding="utf-8")) if gpath.exists() else {"goalies": []}
    skaters = {r["id"]: r for r in rows}
    goalies = {g["id"]: g for g in gdata["goalies"]}
    cats = [c for c in (rows[0]["modes"]["5v5"]["cats"] if rows else {}) if c != "Competition"]

    # team metrics: categories by ice time, goaltending by games
    def profile(abbr: str) -> dict:
        out = {}
        for c in cats:
            num = den = 0.0
            for r in rows:
                v = r["modes"]["5v5"]["cats"].get(c)
                if abbr in _teams_of(r["team"]) and v and v[0] is not None:
                    w = r["gp"] * r["toi"]; num += v[0] * w; den += w
            out[c] = num / den if den else None
        num = den = 0.0
        for g in gdata["goalies"]:
            if abbr in _teams_of(g["team"]) and g.get("overall") is not None:
                num += g["overall"] * g["gp"]; den += g["gp"]
        out["Goaltending"] = num / den if den else None
        return out
    # every metric's value per team, then each team's rank on each, and the overall from the ranks
    now = {_key(t["teamFullName"]): t for t in _summary(y)}
    values = {}
    for t in table:
        abbr, s = t["teamAbbrev"]["default"], now.get(_key(t["teamName"]["default"])) or {}
        values[abbr] = {label: s.get(k) for k, label in TEAM_STATS} | profile(abbr)
    metrics = list(next(iter(values.values())))
    ranks = {m: _ranks({a: v[m] for a, v in values.items()}) for m in metrics}
    mean_rank = {a: (sum(r) / len(r) if (r := [ranks[m][a] for m in metrics if ranks[m][a] is not None]) else None)
                 for a in values}
    overall = _ranks(mean_rank, lower_better=True)

    folder = out / "data" / "teams"
    folder.mkdir(parents=True, exist_ok=True)
    note = (f"Overall: the average of GSAx/GP, SV% and GA/GP percentiles, curved against the {gdata.get('pool', 0)} goalies "
            f"with {gdata.get('min_gp', 0)}+ games.") if gdata["goalies"] else ""
    index = []
    for t in sorted(table, key=lambda t: t["teamName"]["default"]):
        abbr, name = t["teamAbbrev"]["default"], t["teamName"]["default"]
        lp = LINEUPS / f"{abbr}.json"
        saved = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
        combos, roster = saved.get("combinations") or {"players": []}, saved.get("roster") or []
        # match against the roster and everyone who has played for the team this season: the NHL's current roster
        # leaves out players on long-term injured reserve, who Daily Faceoff still lists
        on_roster = {pid for _, pid, _ in roster}
        played = [[r["name"], r["id"], r["pos"]] for r in rows if abbr in _teams_of(r["team"]) and r["id"] not in on_roster]
        played += [[g["name"], g["id"], "G"] for g in gdata["goalies"] if abbr in _teams_of(g["team"]) and g["id"] not in on_roster]
        match = _matcher(roster + played)

        def card(pid, name, slot):
            r, g = skaters.get(pid), goalies.get(pid)
            c = {"id": pid, "name": name, "slot": slot}
            if g and _group(slot) == "G" or g and not r:
                m = {x["key"]: x for x in g["metrics"]}
                sv = m.get("sv", {}).get("value")
                return c | {"pos": "G", "ovr": g.get("overall"), "gp": g["gp"], "rec": f'{g["w"]}-{g["l"]}-{g["otl"]}',
                            "sv": f"{sv:.3f}".lstrip("0") if sv is not None else ""}
            if r:
                o = r["modes"]["5v5"]["overall"]
                return c | {"pos": r["pos"], "ovr": o[0] if o else None, "gp": r["gp"], "g": r["g"], "a": r["a"],
                            "pts": r["g"] + r["a"]}
            return c | {"pos": "G" if _group(slot) == "G" else slot, "ovr": None}

        # even-strength slots carry real positions, so they match reliably; special-teams and IR slots reuse them
        ev = {p["name"]: match(p["name"], p["positionIdentifier"]) for p in combos["players"]
              if p["positionIdentifier"].lower()[:2] not in ("sk", "ir")}
        groups: dict[str, list] = {}
        for p in combos["players"]:
            pid = ev.get(p["name"]) or match(p["name"], p["positionIdentifier"])
            groups.setdefault(p["groupIdentifier"], []).append(card(pid, p["name"], p["positionIdentifier"].upper()))
        lines = [groups[f"f{i}"] for i in range(1, 5) if groups.get(f"f{i}")]
        pairs = [groups[f"d{i}"] for i in range(1, 4) if groups.get(f"d{i}")]
        dressed = {c["id"] for grp in [*lines, *pairs, groups.get("g", [])] for c in grp}
        injured = groups.get("ir", [])
        hurt = {c["id"] for c in injured}
        extras = [card(pid, n, pos) for n, pid, pos in roster if pid not in dressed and pid not in hurt]

        # the metrics in order, each with its rank; P%, PP% and PK% with the stat itself
        shown = [{"k": m, "rank": ranks[m][abbr],
                  "value": f"{100 * values[abbr][m]:.1f}%" if m in dict(TEAM_STATS).values() and values[abbr][m] is not None else ""}
                 for m in metrics]
        page = {
            "abbrev": abbr, "name": name, "season": y, "seasonName": season_name([y]),
            "record": f'{t["wins"]}-{t["losses"]}-{t["otLosses"]}', "points": t["points"], "gp": t["gamesPlayed"],
            "rank": overall[abbr], "teams": len(values), "metrics": shown,
            "lineupSource": {"site": "Daily Faceoff", "updated": combos.get("updatedAt"), "via": combos.get("sourceName"),
                             "url": f"https://www.dailyfaceoff.com/teams/{slug(name)}/line-combinations"},
            "lines": lines, "pairs": pairs, "goalies": groups.get("g", []),
            "pp": [groups.get("pp1", []), groups.get("pp2", [])], "pk": [groups.get("pk1", []), groups.get("pk2", [])],
            "injured": injured, "extras": extras, "goalieNote": note,
        }
        (folder / f"{abbr}.json").write_text(json.dumps(page), encoding="utf-8")
        _logo(abbr, out)
        index.append({"abbrev": abbr, "name": name, "record": page["record"], "points": t["points"], "rank": page["rank"]})
    index.sort(key=lambda t: (t["rank"] or len(index) + 1, t["name"]))   # best first
    (folder / "index.json").write_text(json.dumps({"season": y, "seasonName": season_name([y]), "teams": index}), encoding="utf-8")
    log.info("teams: %d team pages", len(index))
    return len(index)
