# Viktor Challenge Starter — Build the Router

Starter kit for the **Viktor Challenge** at the TUM.ai hackathon (Munich, 22–23 Aug 2026).
From real LLM-request logs, build a router that picks the right model for every call —
then prove it works, even though the log shows only the model that ran, and no outputs or token counts.

## Quick start (5 minutes)

```bash
# 1. No dataset yet? Generate a synthetic sample with the same shape:
python scripts/make_synthetic_sample.py            # writes ./export/

# 2. Got the real dataset links (shipped at kickoff)? Then instead: the export ships
#    as trajectories_v1_<index>.jsonl.tar.gz archives — download, verify the posted
#    SHA-256, then:  mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/

# 3. Sanity-check the export, reconstruct trajectories, print stats:
python scripts/load_trajectories.py export/

# 4. Run the baseline heuristic router + cache-aware cost report:
python scripts/baseline_router.py export/

# 5. Turn results into a cost–quality frontier CSV (+ PNG if matplotlib is installed):
python scripts/plot_frontier.py results/routes.jsonl
```

Python 3.10+, standard library only (matplotlib optional for the PNG).

## Routemaxxing

The challenge implementation corrects the unit of analysis before routing: the real
chunk's 1,000 unique rows unfold into 10,845 model turns. It trains an interpretable,
family-preserving downgrade policy and reports a public-validation frontier with a
2-percentage-point proxy-quality constraint:

```bash
python scripts/routemaxxing.py export/
```

The command writes the certified and one-escalation route ledgers, frontier CSV/SVG,
and a fully traceable summary under `results/routemaxxing/`. It also writes
`results/routemaxxing.jsonl` for the local explorer and refreshes
`presentation-routemaxxing.html`. All token and dollar figures are estimates; final
outputs are excluded in the briefing-compatible primary analysis, and adaptive quality
is explicitly conditional on the logged context.

### Live demo

The aggregate-only Router Lab is deployable as a static GitHub Pages site:

```bash
python scripts/build_live_demo.py
python -m http.server 4173 -d docs
```

Open <http://127.0.0.1:4173>. The live deployment is
<https://utkukalkanli.github.io/routemaxxing/>. The page contains the interactive
certified policy, a cache-aware cost simulator, the validation frontier, and required
comparisons. It deliberately ships no challenge trajectories, prompts, tools, or route
ids. `.github/workflows/pages.yml` can deploy `docs/` from this repository after its
owner enables GitHub Pages with GitHub Actions as the source.

## Using a coding agent

Point Claude Code / Codex / Cursor / opencode at this repo — `AGENTS.md` briefs your agent.
In Claude Code you also get slash commands:

- `/setup` — set up everything needed to participate
- `/make-presentation` — build a Viktor-branded presentation of your solution
- `/prepare-submission` — package your solution into a formal submission

## What's here

| Path | What |
|---|---|
| `AGENTS.md` | Agent briefing: dataset shape, the cache trap, judging, starter ideas |
| `skills/` | The three guided workflows above (plain Markdown, readable by humans too) |
| `scripts/` | Loader + trajectory reconstruction, baseline router, cache-aware cost model (estimated tokens), frontier plot, synthetic sample |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
