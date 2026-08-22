# Viktor — TUM.ai Hackathon Combined Deck

*TUM.ai hackathon · Munich · 22–23 August 2026*

---

## 1. Meet Viktor

**The AI employee.**
A frontier-level knowledge work agent. You talk to him in Slack or Microsoft Teams.

---

## 2. Meet Viktor — *always online*

- Viktor lives in Slack and Microsoft Teams. He does tasks and complex work items.
- You talk to Viktor as you talk to a teammate. He keeps context and learns over time.
- Proactive: Viktor explores your Slack and integrations, builds an understanding of **who is responsible for what**, and starts proposing how to automate the whole company, step by step.

**Flow:**
1. **explore** — your Slack + integrations (Slack, Email, CRM, Code, Docs, +3,000 more). *Viktor reads how work actually flows.*
2. **understand** — invoices · finance, leads · sales, deploys · engineering. *who is responsible for what*
3. **automate** — automation proposals (chase overdue invoices, triage inbound leads, weekly KPI report...). *step by step — you approve each*

---

## 3. Traction

| | |
|---|---|
| **55,000+** | companies already run on Viktor *(and counting)* |
| **$75M** | raised already — Viktor launched in February 2026 |
| **5M+** | tool calls he makes every day — real work, not chat |
| **3,200+** | tools and integrations he can use out of the box |

---

## 4. How he works

Viktor runs on his own computer in the cloud, always on. He does not follow fixed workflows. He writes and runs code.

1. **Code-first harness** — Every tool is a Python SDK function. Viktor writes scripts to do the work. A smarter model gets more out of the same harness.
2. **Shared filesystem** — Scripts and data persist across sessions. A script that worked once becomes a reusable tool.
3. **Skills as memory** — He stores what he learns in skill files. He loads them only when a task needs them. The tenth task is better than the first.* *(footnote: the rules for making this deck live in a skill file, too.)*
4. **Proactive loop** — A heartbeat and scheduled tasks let him pull work like a colleague. Subagents fan out large jobs. A human approves the result.

> **One test for each harness decision:** is it useful for the next model generation? If yes, build it.

---

## 5. Model Router *(challenge intro)*

Train a router that picks the right model for each LLM call. You set the objective.

Viktor is the AI employee in Slack and Microsoft Teams. Each of his tasks is a sequence of LLM calls.

Real agent traces, with an open off-policy problem inside.

*Viktor challenge track · Munich · 22–23 August 2026*

---

## 6. The task

Input: real Viktor trajectories (model, input, tools per call). Open-ended — team selects and defends the objective.

- **(A) Build the router** — assign a model per call; optimize cost, quality, or your own trade-off.
- **(B) Prove it works** — the log shows only the model that ran; estimate what your router changes.
- Ship: a router + a defensible evaluation + a short presentation.

**NO GPU NEEDED** — weekend-sized static offline export, baselines train on a laptop in minutes, no GPU/API key. The depth is in the off-policy evaluation, not in compute.

---

## 7. The dataset

Static offline export, chunked JSONL, one line per real LLM request. OpenAI-compatible Responses format, exactly three fields: `model`, `input`, `tools`.

- ~9 model ids across families, e.g. `claude-opus-5`, `claude-fable-5`, `gpt-5.6-terra` — **you infer the tier order** (no official price sheet at deck time).
- No output, no usage, no trajectory ids. *We freeze the export before kickoff.*
- Layout: `export/trajectories_v1_01.jsonl`, `_02.jsonl`, ... — one line = one request. No ids — rebuild tasks by grouping on opening messages.

---

## 8. A trajectory

- **`model`** — the id that served the call; one model serves the whole trajectory.
- **`input`** — full request history: message, function_call, function_call_output, reasoning items; contains every call before it.
- **`tools`** — function defs available to the call (the real Viktor toolset).
- **`output` / `usage`** — not shipped. What call *i* returned appears inside call *i+1*'s input — only the final output is truly gone. Token counts must be estimated by the team.

**Diagram:** `input (user message)` → goes into → `call_1 (output: function_call)` → runs → `function output appended to input` → input grows → `call_2 (output: message)`.
*Caption: many calls, one model, one growing input prefix — which calls needed that model?*

---

## 9. Sample

Abridged real line: `claude-opus-5` call with a system prompt ("You are Viktor Ai (@viktor in slack…)"), a user message about a "payment brief posted to [person_1] DM", a `wait_for_background_work` function_call/output pair, 18 more items, and a `tools` array (bash + 12 tools).

> No `output`. No `usage`. The next request of this task repeats this whole input — plus what the model did. Remember this.

---

## 10. The signal

**What you observe per call:** input items (messages/function calls/outputs/reasoning stubs, each call embedding every prior call), tools, and the model's prior actions (recoverable from call *i+1*'s input — only the final task output is missing). `model` — one id per trajectory, varies across trajectories: that variation is your natural experiment.

**What does not ship:** no quality label, no timing, no output/usage. Every token count is a team estimate, to be labeled as such.

The team must construct its own outcome signal — options given: judge-model scores, task completion read from the final message, function-call error patterns.

One model serves all calls of a trajectory. Which calls needed it? That question is the twist (next slide). No routing label and no quality label ship — constructing a good signal is part of the challenge; a good metric is a result in itself.

---

## 11. The twist — off-policy evaluation

**ONE TRAJECTORY — FIVE CALLS.** Diagram: `call_1` … `call_5`, all served by `gpt-5`. *One model served all five calls: gpt-5. Each input contains the calls before it, served from the cache.*

Below it, a hypothetical routed alternative row (`gpt-5-mini?`, `gpt-5`, `claude-haiku?`, `gpt-5-mini?`, `gpt-5`). *The routed alternative — the log never shows this row. Each model switch resets the cache.*

Key questions: would a cheaper model have kept the result on some calls? The log doesn't show it.

Cost has a second trap. Providers cache the shared input prefix. A model switch resets that cache. Calls after a switch pay the full price. Estimate the two effects.

**The best estimate wins its own prize.**

Starting points given:
- compare calls across trajectories on different models
- judge-model rescoring
- cache-aware repricing from the input prefix a call shares with the one before it (no usage field ships)

---

## 12. Evaluation

Open-ended — team defines objective + evaluation and defends both. Held-out split ships with the dataset at kickoff.

- **Cost–quality frontier** — for the chosen cost metric (tokens or $), show what quality the router keeps at what cost, priced cache-aware (cached input tokens cheap; a model switch resets the cache). *The frontier is the headline, not a single point.*
- **Off-policy honesty** — log shows one model per trajectory; state the estimation method (matching, weighting, judge models) and where it can fail.
- How it is judged — three criteria, next slide.

**Chart** (outcome quality vs. cost per task): a "your router" curve rising above a dashed "random routing" line, reaching an "always the frontier model" point at top-right, with a "be here" arrow pointing to a spot on the curve below that point.
*Caption: same quality, lower cost — best frontier wins.*

---

## 13. Judging (3 criteria)

1. **Routing insight** — does the router use real, nameable structure in the traces? *A signal someone can point at, not a black box with a good score.*
2. **Evaluation rigor** — is the off-policy estimate honest, given one model per trajectory and a cache that couples the calls? *You name where your estimate can fail before the panel does.*
3. **Presentation clarity** — defend the frontier in 5 minutes. *One chart, one claim, one known weakness.*

Workflow: build on public split → final run on held-out split → show cost–quality frontier → defend method to panel.

**Criteria weights announced at kickoff.** Special prize (best off-policy-evaluation insight) cuts across all three.

---

## 14. Get started — first hour, in order

1. Join challenge Discord: `discord.gg/J5eNgaAT` (Viktor team answers there all weekend).
2. Download the starter kit (link shared at kickoff briefing, pinned in Discord).
3. Download the dataset from the kickoff link: `tar xzf trajectories_v1_01.jsonl.tar.gz -C export/`
4. Run `python scripts/load_trajectories.py export/` then `python scripts/baseline_router.py export/` → reconstructed trajectories + a baseline frontier. No GPU, no API key.
5. Point a coding agent at the kit — `AGENTS.md` briefs it. Build the router, beat the baseline.

---

## 15. Starter ideas

- **idea/01 (the safe bet) — Heuristic router:** route by prompt length and call position; ~1 hour gives an honest cost floor against logged behavior.
- **idea/02 (the ambitious one) — Learned router:** construct an outcome signal, compare similar tasks that ran on different models, train a classifier that picks the cheapest model keeping the outcome.
- **idea/03 (the special prize) — Honest evaluation:** match or weight trajectories run on different models; rescore single calls with a judge model. *(This card is "the map" the judging slide points to for the special prize.)*
- **idea/04 (check before you celebrate) — Cache-aware costing:** reprice each route with cached/uncached token rates, inferring cached share from the shared input prefix with the previous call; show how much a naive estimate overstates savings.

---

## 16. Rules & prizes

- **Format** — team size, submission format, deadline follow general TUM.ai hackathon rules, confirmed at kickoff briefing.
- **Compute** — track's compute/API credits announced at kickoff; dataset itself runs on a laptop.
- **License** — do not use the data outside this challenge, do not redistribute it.
- Framed as a portfolio piece either way (routing + off-policy eval on real agent traces) plus hiring conversations: `viktor.com/jobs`.

**Prizes** (on top of base TUM.ai prizes):
- 1st — $2,000 Viktor workspace credits **+ a day trip to the Warsaw office**
- 2nd — $1,000 credits
- 3rd — $500 credits
- Special prize for best off-policy-evaluation insight (maps to slide 15 card 03)

---

## 17. Work with us

Hiring engineers to build agents that do real work — small team, high ownership, production AI from day one.

Open roles: `viktor.com/jobs`.

---

## 18. Contact

> One dataset, one weekend, one question: which model should have taken this call?

- Mentors on-site all weekend: Bojan and Peter; Ivan mentors remotely. Table location/office hours announced at kickoff.
- Challenge Discord: `discord.gg/J5eNgaAT` — assets, direct Viktor-team access, a pre-configured Viktor instance to try.
- Dataset download link shared at kickoff briefing.
- Careers: `viktor.com/jobs`.

*P.S. — Viktor built this deck himself. He is curious what you build.*
*P.P.S. — He left one more message in the browser console.*
*(Note: the PDF itself just says this — it doesn't show the console message's actual text. That content, if captured, would only be visible by opening the live page's dev console, not from this PDF.)*
