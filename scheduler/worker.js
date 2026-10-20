// Cloudflare Worker that starts the "Update site" GitHub Actions workflow on a cron.
// GitHub's own schedule is best-effort and runs a 5-minute cron only every few hours;
// Cloudflare's cron triggers run on time. Each run that finds nothing new takes about
// a minute and deploys nothing, and the workflow's concurrency group means a run that
// is still going keeps at most one more queued behind it.
//
// Secret: GITHUB_TOKEN, a fine-grained token for this repo with Actions: Read and write.

const REPO = "CascV8/nhl-microstats";
const WORKFLOW = "update-site.yml";

async function dispatch(env) {
  const res = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "nhl-microstats-scheduler",
    },
    body: JSON.stringify({ ref: "main" }),
  });
  if (res.status !== 204) throw new Error(`GitHub answered ${res.status}: ${await res.text()}`);
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(dispatch(env));
  },
  // Only reachable if the worker is given a URL (it has none by default); never triggers a run.
  async fetch() {
    return new Response(`Starts ${WORKFLOW} in ${REPO} every 5 minutes.\n`);
  },
};
