// NHLe cards: the JS model against the NHLe repo's Python models, and the API normalizers.
// Run with: node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { buildCard, firstEligibleDraft, leagueKey, positionGroup, probabilities } from "../src/nhl_microstats/nhle.js";
import { epPlayer, nhlPlayer, onRequestGet, startYear } from "../functions/api/[[path]].js";

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
const landing = JSON.parse(readFileSync(new URL("./fixtures/nhl_landing_bedard.json", import.meta.url), "utf-8"));

test("probabilities match the Python models", () => {
  // Values from NHLe/Forwards/fwd-d*.py and Defense/d-d0.py (p_nhler, p_star).
  let p = probabilities("F", "D+1", 61 / 68 * 82);
  close(p.nhler, 0.9998030293651083); close(p.star, 0.8999460966013588);
  p = probabilities("F", "D+2", 67);
  close(p.nhler, 0.9983810868096824); close(p.star, 0.8861561092424858);
  p = probabilities("D", "DY", 10);
  close(p.nhler, 0.660743218074922); close(p.star, 0.1871477419189746);
  // Star never exceeds NHLer.
  p = probabilities("F", "D+5", 0);
  assert.ok(p.star <= p.nhler);
});

test("Bedard's card pools leagues, playoffs and the right World Juniors", () => {
  const player = nhlPlayer(landing);
  assert.equal(player.name, "Connor Bedard");
  assert.deepEqual(player.draft, { year: 2023, round: 1, overall: 1, team: "CHI" });
  const card = buildCard(player, 2026);
  assert.equal(card.group, "F");
  assert.deepEqual(card.seasons.map(s => s.year), [2021, 2022, 2023, 2024, 2025, 2026, 2027]);

  const [d1, dy, dp1, dp2, dp3, dp4, dp5] = card.seasons;
  // D-1 (2021-22): WHL + WJC-18; his WJC-20 games don't count in the D-1 model.
  close(d1.pts82, 26.06027282975454); close(d1.nhler, 0.9439406428393012); close(d1.star, 0.6235173917652341);
  assert.equal(d1.rows.find(r => r.league === "WJC-20").used, false);
  // Draft year: WHL regular season + playoffs + WJC-20.
  close(dy.pts82, 44.687005191726755); close(dy.nhler, 0.9994552993063687); close(dy.star, 0.9396980415739763);
  // D+1 is NHL only (the WC and "International" have no factor).
  close(dp1.pts82, 61 / 68 * 82);
  assert.equal(dp1.gp, 68);
  // 150 NHL games through D+2, 219 through D+3: NHLer becomes a known outcome from D+3.
  assert.ok(!dp2.known && dp2.nhler < 1);
  assert.ok(dp3.known); assert.equal(dp3.nhler, 1);
  assert.ok(dp4.inProgress && dp4.played);
  assert.ok(dp5.future && !dp5.played);
  assert.equal(card.latest, dp4);
});

test("duplicate NHL stints are dropped", () => {
  const dup = { ...landing, seasonTotals: [...landing.seasonTotals, landing.seasonTotals.at(-1)] };
  assert.equal(nhlPlayer(dup).stats.length, nhlPlayer(landing).stats.length);
});

test("league names from the NHL and Elite Prospects map to model keys", () => {
  for (const [name, key] of [["WHL", "whl"], ["HockeyAllsvenskan", "allsvenskan"], ["J20 Nationell", "u20-nationell"],
    ["j20-nationell", "u20-nationell"], ["U20 SM-liiga", "u20-sm-sarja"], ["Czechia U20", "czechia-u20"], ["U20-Elit", "u21-elit"],
    ["WJC-20", "wjc-20"], ["Liiga", "liiga"], ["NHL", "nhl"]])
    assert.equal(leagueKey(name), key, name);
  for (const name of ["ECHL", "WJC-20 D1A", "U18 SM-sarja", "16U AAA", "", null]) assert.equal(leagueKey(name), null, name);
});

test("positions and draft eligibility", () => {
  assert.equal(positionGroup("C"), "F"); assert.equal(positionGroup("LW/RW"), "F");
  assert.equal(positionGroup("D"), "D"); assert.equal(positionGroup("G"), null);
  assert.equal(firstEligibleDraft("2005-07-17"), 2023);   // Bedard
  assert.equal(firstEligibleDraft("2007-09-15"), 2025);   // 18 on the cut-off
  assert.equal(firstEligibleDraft("2007-09-16"), 2026);
});

test("an undrafted prospect is placed by first eligible draft", () => {
  const card = buildCard({ name: "X", position: "D", birthDate: "2009-03-01", draft: null,
    stats: [{ season: 2025, type: "reg", league: "ohl", team: "T", gp: 60, g: 10, a: 30, pts: 40 },
            { season: 2025, type: "reg", league: "U18 AAA", team: "T", gp: 5, g: 9, a: 9, pts: 18 }] }, 2026);
  assert.equal(card.draftYear, 2027);
  assert.equal(card.drafted, false);
  assert.equal(card.seasons[0].year, 2025);   // D-1
  close(card.seasons[0].pts82, 40 * 0.1767277524 / 60 * 82);   // the unrated league is left out
});

test("Elite Prospects rows normalize to the same shape", () => {
  const p = epPlayer({ id: 1, name: "A B", dateOfBirth: "2008-01-02", position: "C", nationality: { name: "Sweden" } },
    [{ season: { slug: "2025-2026" }, league: { slug: "j20-nationell" }, teamName: "Frölunda HC J20",
       regularStats: { GP: 40, G: 20, A: 25, PTS: 45 }, postseasonStats: { GP: 4, G: 1, A: 2, PTS: 3 } },
     { season: { slug: "2025-2026" }, league: { slug: "shl" }, teamName: "Frölunda HC", regularStats: { GP: 0 } }],
    [{ year: 2026, round: 1, overall: 5, team: { name: "Team" }, draftType: { slug: "nhl-entry-draft" } }]);
  assert.equal(p.stats.length, 2);   // a season with no games is dropped
  assert.deepEqual(p.stats[1], { season: 2025, league: "j20-nationell", team: "Frölunda HC J20", type: "po", gp: 4, g: 1, a: 2, pts: 3 });
  assert.equal(p.draft.overall, 5);
  assert.equal(startYear(20222023), 2022); assert.equal(startYear("2022-23"), 2022); assert.equal(startYear({ startYear: 2019 }), 2019);
});

test("the API rejects bad parameters without calling upstream", async () => {
  const call = async (path, qs, env = {}) => {
    const res = await onRequestGet({ request: new Request(`https://x.dev/api/${path}?${qs}`), env, params: { path: [path] } });
    return [res.status, await res.json()];
  };
  assert.deepEqual(await call("sources", ""), [200, { ep: false }]);
  assert.equal((await call("player", "src=nhl&id=../../etc"))[0], 400);
  assert.equal((await call("draft", "year=20x5"))[0], 400);
  assert.equal((await call("league", "league=ohl&season=2025"))[0], 400);   // needs Elite Prospects
  assert.equal((await call("nope", ""))[0], 404);
  assert.deepEqual(await call("search", "q=a"), [200, []]);
});
