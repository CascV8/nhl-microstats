// NHLe prospect model: a player's NHL-equivalent scoring, season by season from the year before
// the draft (D-1) to five years after it (D+5), and what each season says about his chances.
// A port of the NHLe repo's per-season models (Forwards/fwd-d*.py, Defense/*.py); the coefficients
// are copied from there. Shared by the NHLe page (in the browser) and the tests (in node).
//
// Each season: pts/82 = sum(points x league factor) / sum(games) x 82, pooling regular season,
// playoffs and the season's eligible World Juniors. Then
//   P(NHLer) = sigmoid(a + b * pts/82)    reaches 200 NHL games
//   P(Star)  = min(sigmoid(c + d * pts/82), P(NHLer))    top 15% peak GAR among NHL forwards

export const SEASONS = ["D-1", "DY", "D+1", "D+2", "D+3", "D+4", "D+5"];

// NHL-equivalency factors, keyed by Elite Prospects league slug (the NHLe repo's keys).
export const FACTORS = {
  "nhl": 1.0, "shl": 0.6645193365, "liiga": 0.6285014507, "khl": 0.555534104, "czechia": 0.523293024,
  "nl": 0.4934002709, "del": 0.3957446585, "ahl": 0.3895545856, "vhl": 0.3667523758, "slovakia": 0.3567670587,
  "ncaa": 0.3179628572, "allsvenskan": 0.3070513875, "wjc-20": 0.305254264, "wjc-18": 0.2207053337,
  "sl": 0.2426592162, "mestis": 0.228940698, "del2": 0.2269787804, "latvia": 0.2112029502, "whl": 0.1943040421,
  "czechia2": 0.1824508245, "ohl": 0.1767277524, "u20-sm-sarja": 0.1695791393, "qmjhl": 0.1673133115,
  "mhl": 0.152009593, "u20-nationell": 0.1519723109, "ushl": 0.1726835286, "czechia-u20": 0.1037596731,
  "bchl": 0.09306532439, "slovakia2": 0.0915762792, "u21-elit": 0.04983498016, "dnl": 0.02992688863,
};

// Other spellings of those leagues: NHL API abbreviations and Elite Prospects slugs/names, lowercased.
const ALIASES = {
  "hockeyallsvenskan": "allsvenskan", "j20-nationell": "u20-nationell", "j20-superelit": "u20-nationell",
  "u20-sm-liiga": "u20-sm-sarja", "jr.-a-sm-liiga": "u20-sm-sarja", "czech": "czechia", "czech2": "czechia2",
  "czech-u20": "czechia-u20", "czechia-u19": "czechia-u20", "extraliga": "czechia", "u20-elit": "u21-elit",
  "elite-jr.-a": "u21-elit", "dnl-u20": "dnl", "national-league": "nl", "swiss-league": "sl", "wjc20": "wjc-20",
  "wjc18": "wjc-18", "wjc-u20": "wjc-20", "wjc-u18": "wjc-18", "big10": "ncaa", "ncaa-d1": "ncaa",
};

/** The model's key for a league name or slug, or null when it has no NHLe factor. */
export function leagueKey(name) {
  const k = String(name || "").trim().toLowerCase().replace(/\s+/g, "-");
  if (k in FACTORS) return k;
  return ALIASES[k] || null;
}

// Coefficients per draft-relative season. wjc: tournaments pooled into that season;
// known: the model turns P(NHLer) to 100% once the player has 200 NHL games.
const F = {
  "D-1": { nhler: [-3.198512, 0.231086], star: [-3.553653, 0.155722], wjc: ["wjc-18"] },
  "DY":  { nhler: [-3.158896, 0.238853], star: [-4.432684, 0.160648], wjc: ["wjc-18", "wjc-20"] },
  "D+1": { nhler: [-2.335101, 0.147737], star: [-2.900780, 0.069297], wjc: ["wjc-20"] },
  "D+2": { nhler: [-2.637906, 0.135258], star: [-3.220768, 0.078699], wjc: ["wjc-20"], known: true },
  "D+3": { nhler: [-2.731481, 0.213728], star: [-3.757927, 0.079313], wjc: [], known: true },
  "D+4": { nhler: [-3.789709, 0.260738], star: [-4.426382, 0.095628], wjc: [], known: true },
  "D+5": { nhler: [-3.472162, 0.226039], star: [-7.221153, 0.205414], wjc: [], known: true },
};
// Defensemen have their own D-1 and draft-year models; from D+1 on the NHLe repo uses the forward ones.
export const MODELS = {
  F,
  D: { ...F,
    "D-1": { nhler: [-2.576679, 0.148630], star: [-3.439546, 0.117490], wjc: ["wjc-18"] },
    "DY":  { nhler: [-1.802312, 0.246892], star: [-3.395801, 0.192715], wjc: ["wjc-18", "wjc-20"] } },
};

const TOURNAMENTS = new Set(["wjc-18", "wjc-20"]);
const sigmoid = x => 1 / (1 + Math.exp(-x));

export function probabilities(group, label, pts82) {
  const m = MODELS[group][label];
  const nhler = sigmoid(m.nhler[0] + m.nhler[1] * pts82);
  return { nhler, star: Math.min(sigmoid(m.star[0] + m.star[1] * pts82), nhler) };
}

/** "F" or "D" from any position code (C, LW, R, D, "C/LW", ...); null for goalies. */
export function positionGroup(pos) {
  const p = String(pos || "").toUpperCase();
  if (p.startsWith("G")) return null;
  return p.split("/")[0] === "D" ? "D" : "F";
}

/** First draft a player is eligible for: 18 by September 15 of the draft year. */
export function firstEligibleDraft(birthDate) {
  const [y, m, d] = String(birthDate).split("-").map(Number);
  return y + 18 + (m > 9 || (m === 9 && d > 15) ? 1 : 0);
}

/**
 * The card for one player. ``player`` is the normalized shape the /api/player endpoint returns:
 *   { name, position, birthDate, draft: {year, round, overall, team} | null,
 *     stats: [{ season: start year, type: "reg" | "po", league, team, gp, g, a, pts }] }
 * ``current`` is the start year of the season under way (later seasons are "not yet played").
 */
export function buildCard(player, current) {
  const group = positionGroup(player.position);
  if (!group) return { group: null, seasons: [] };
  const draftYear = player.draft?.year || firstEligibleDraft(player.birthDate);
  const dy = draftYear - 1;   // the draft-year season starts the autumn before the draft
  const nhlGpThrough = y => player.stats.filter(s => s.season <= y && s.type === "reg" && leagueKey(s.league) === "nhl")
    .reduce((t, s) => t + s.gp, 0);

  const seasons = SEASONS.map((label, i) => {
    const year = dy - 1 + i, model = MODELS[group][label];
    const rows = player.stats.filter(s => s.season === year).map(s => {
      const key = leagueKey(s.league);
      const used = key != null && s.gp > 0 && (!TOURNAMENTS.has(key) || model.wjc.includes(key));
      return { ...s, key, factor: key ? FACTORS[key] : null, used };
    });
    const used = rows.filter(r => r.used);
    const gp = used.reduce((t, r) => t + r.gp, 0);
    const out = { label, year, rows, gp, played: gp > 0, future: year > current, inProgress: year === current };
    if (!out.played) return out;
    out.nhlePts = used.reduce((t, r) => t + r.pts * r.factor, 0);
    out.pts82 = out.nhlePts / gp * 82;
    Object.assign(out, probabilities(group, label, out.pts82));
    if (model.known && nhlGpThrough(year) >= 200) { out.nhler = 1; out.known = true; }
    return out;
  });
  const latest = [...seasons].reverse().find(s => s.played);
  return { group, draftYear, drafted: !!player.draft?.year, seasons, latest };
}
