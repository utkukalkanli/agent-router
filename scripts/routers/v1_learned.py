#!/usr/bin/env python3
"""idea/02 (the ambitious one) — Learned router.

Stdlib-only: no sklearn, no API key. Builds an outcome signal from the logged
trajectories themselves, buckets trajectories by coarse structural features to
find "similar tasks that ran on different models" (the dataset has no exact
task run twice under different models — see AGENTS.md/deck slide 10's "natural
experiment" framing), and for each bucket routes to the cheapest observed model
that kept the outcome within a threshold of the bucket's best model.

Caveat (honesty, matches idea/03's spirit): this fits on the same public split
it then routes — it has not been validated against a held-out split. Re-run
fit_bucket_routes once the held-out split ships at kickoff before trusting the
frontier this produces.

Usage: python scripts/routers/v1_learned.py export/
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_trajectories import iter_requests, group_trajectories, est_tokens  # noqa: E402
from cost_model import load_pricing, price_of  # noqa: E402
from _common import write_routes  # noqa: E402

ERROR_MARKERS = re.compile(
    r"\berror\b|\bfailed\b|\bexception\b|\btraceback\b|\bdenied\b|\binvalid\b|"
    r"\btimeout\b|\brefused\b|\bnot found\b|\bunauthorized\b", re.IGNORECASE)
HEDGE_MARKERS = re.compile(r"\bsorry\b|\bunable\b|\bcouldn't\b|\bcannot\b|\bcan't\b|\bnot able\b", re.IGNORECASE)

def _tercile_edges(values):
    """(lo, hi) cut points splitting sorted `values` into thirds. Data-driven —
    fixed thresholds guessed ahead of time turned out to not match the real
    dataset's scale at all (e.g. every trajectory here has >=9 tools, so a
    fixed "few tools <= 3" edge classified everything as "many" and collapsed
    the whole bucket space to 5 buckets for 953 trajectories)."""
    values = sorted(values)
    n = len(values)
    lo = values[min(n // 3, n - 1)]
    hi = values[min((2 * n) // 3, n - 1)]
    return (lo, hi)


def compute_edges(groups):
    """Tercile edges for tokens / tools / call-count, computed from the actual
    trajectories being routed (not hardcoded) so bucketing adapts to whatever
    export it's run against."""
    calls_list = list(groups.values())
    toks = [est_tokens(c[-1]["input"]) for c in calls_list]
    tools = [len(c[-1].get("tools") or []) for c in calls_list]
    ncalls = [len(c) for c in calls_list]
    return {"tokens": _tercile_edges(toks), "tools": _tercile_edges(tools), "calls": _tercile_edges(ncalls)}


def _band(value, edges, labels):
    lo, hi = edges
    if value <= lo:
        return labels[0]
    if value <= hi:
        return labels[1]
    return labels[2]


def bucket_key(calls, edges):
    """Coarse, nameable structure a judge can point at: prompt size x tool count
    x trajectory length — the proxy for "similar task" since no exact task is
    ever logged twice under two different models. `edges` comes from
    compute_edges(groups) so the bands reflect this dataset's real spread."""
    last = calls[-1]
    tok = est_tokens(last["input"])
    n_tools = len(last.get("tools") or [])
    n_calls = len(calls)
    return (_band(tok, edges["tokens"], ("small", "medium", "large")),
            _band(n_tools, edges["tools"], ("few", "some", "many")),
            _band(n_calls, edges["calls"], ("short", "medium", "long")))


def _text_of(item):
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def outcome_score(calls):
    """Heuristic outcome signal in [0, 1] from the LOGGED trajectory only.
    0.6 completion + 0.4 (1 - function_call error rate). The final call's real
    output is never in the export (permanently missing) -- this only reads
    what's recoverable: every output up to call N-1, embedded in call N's input."""
    items = calls[-1]["input"]
    last = items[-1] if items else None
    if last and last.get("type") == "message" and last.get("role") == "assistant":
        completion = 0.5 if HEDGE_MARKERS.search(_text_of(last)) else 1.0
    else:
        completion = 0.0

    outputs = [it for it in items if it.get("type") == "function_call_output"]
    if outputs:
        errors = sum(1 for o in outputs if ERROR_MARKERS.search(str(o.get("output", ""))))
        error_component = 1.0 - errors / len(outputs)
    else:
        error_component = 1.0  # no function calls -> no evidence of error

    return round(0.6 * completion + 0.4 * error_component, 6)


def fit_bucket_routes(groups, pricing, threshold_type="relative", threshold_value=0.98, min_samples=3, edges=None):
    """{bucket: cheapest_model} learned from cross-trajectory comparison within
    each bucket. threshold_type is "relative" (fraction of the bucket's best
    score) or "absolute" (a fixed score floor). A model needs at least
    `min_samples` observed trajectories in a bucket before it's trusted as a
    routing candidate -- one lucky clean trajectory isn't evidence. If no
    model in a bucket clears that bar, the bucket is left out of the result
    entirely (callers fall back to each trajectory's own logged model). If
    models clear the sample bar but none clears the score threshold, falls
    back to the single best-scoring reliable model. `edges` defaults to
    compute_edges(groups) -- pass the same edges used here into
    make_route_trajectory so fit and apply agree on bucket boundaries."""
    edges = edges or compute_edges(groups)
    bucket_scores = {}  # bucket -> {model: [scores]}
    for calls in groups.values():
        b = bucket_key(calls, edges)
        model = calls[0]["model"]
        bucket_scores.setdefault(b, {}).setdefault(model, []).append(outcome_score(calls))

    routes = {}
    for b, per_model in bucket_scores.items():
        reliable = {m: s for m, s in per_model.items() if len(s) >= min_samples}
        if not reliable:
            continue
        avg = {m: sum(s) / len(s) for m, s in reliable.items()}
        best_score = max(avg.values())
        floor = best_score * threshold_value if threshold_type == "relative" else threshold_value
        candidates = [m for m, s in avg.items() if s >= floor]
        if not candidates:
            candidates = [max(avg, key=avg.get)]
        routes[b] = min(candidates, key=lambda m: price_of(m, pricing)[0])
    return routes


def make_route_trajectory(bucket_routes, edges):
    """`edges` MUST be the same edges fit_bucket_routes used -- otherwise a
    trajectory could be bucketed differently at apply time than it was during
    fitting."""
    def route_trajectory(calls):
        model = bucket_routes.get(bucket_key(calls, edges), calls[0]["model"])
        return [model] * len(calls)
    return route_trajectory


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    pricing = load_pricing()
    groups = group_trajectories(r for _, _, r in iter_requests(export))  # pass 1: fit
    edges = compute_edges(groups)
    bucket_routes = fit_bucket_routes(groups, pricing, edges=edges)
    write_routes(make_route_trajectory(bucket_routes, edges), export=export, name="v1_learned")  # pass 2: apply + write


if __name__ == "__main__":
    main()
