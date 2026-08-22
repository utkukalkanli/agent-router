# Local trajectory explorer

Python 3.10+ is required. Install the local dependencies once, then start the
indexer and web server with one command:

```bash
python -m pip install -r requirements.txt
python -m explorer
```

Open <http://127.0.0.1:8000>. The process reads only local `export/*.jsonl` and
`results/*.jsonl` files. It makes no telemetry, analytics, CDN, font, or other
network requests.

For frontend development:

```bash
cd explorer/frontend
npm install
npm run build
```

The production bundle is committed under `explorer/static/`, so Node is not
needed to run the explorer. Dataset rows are never included in that bundle.

## What the tabs show

| Tab | What it answers |
|---|---|
| **Overview** | What the corpus costs, which models spend it, how much of it is the uncached cold start, and how concentrated it is. Also states plainly whether `results/*.jsonl` joined. |
| **Classes** | Ten ways the inputs sort themselves — workload, trigger, harness, chat surface, toolset, size, depth, cron family, loop shape, model. All read out of the requests, none guessed. Click a row to filter the trajectory list. |
| **Function calls** | Each tool ranked by the tokens its *result* adds to every later prompt — i.e. which calls make a run expensive to continue, not which are called most. |
| **Trajectories** | Faceted list of runs. Open one for its Flow: call by call, each request a bar split cached / new / output, with per-call cost, tool chips and expandable item text. |
| **Compare** | Nearest runs by tool overlap and loop shape, then an LCS alignment of two runs showing which steps both took and where the cost gap sits. Plus matched cohorts as an off-policy estimate, with the size-spread confounder exposed. |
| **Files** | `export/` and `results/` line by line: what each file contributed, each row's decomposition, and the raw JSON. Jump from a raw line to its Flow. |
| **Raw calls** | The unopinionated 50-column table over the `calls` index. |

## Reading the numbers

Every token count is an **estimate** — serialized JSON chars / 4. The export ships no
`usage` field, so nothing here is measured. Outputs are recovered from the next call's
input, which leaves the final turn of every trajectory unobservable. Costs are
cache-aware and input+output, at the official rates in `scripts/cost_model.py`.

One export line is one **complete trajectory**, not one call: the agent loop ran inside
it, and `scripts/decompose.py` unfolds it into the ~10.8k requests that actually ran.
Trajectory ids are that decomposition's ids. A router artifact in `results/` only shows
up in the cost columns if its ids were minted the same way — the Overview tab reports
whether they were.
