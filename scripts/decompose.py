#!/usr/bin/env python3
"""Unfold each export row into the LLM calls that actually ran, and price them.

WHY THIS EXISTS
---------------
`load_trajectories.py` assumes one row = one LLM call and reconstructs trajectories by
grouping rows. On the real export that finds nothing: every row has a unique first user
message, and the item-level prefix any two rows share is always exactly 1 item (the
system message). See `trajectory_ids.py` for the verification.

The reason is that one row = one COMPLETE trajectory. The agent loop ran inside it:
model emits a tool call -> sandbox returns a result -> the whole history is sent again.
A 19-item row is 7 billed requests, not 1. Those intermediate requests really ran and
were really billed; costing the row as a single call undercounts the corpus by ~3.3x.

It also means the outputs are NOT missing. Everything the model produced is sitting in
the item list between the tool results, and every row ends with the final assistant
message. Decomposing recovers ~11.4k (input -> output) pairs with real output tokens —
the labels the challenge brief says the export does not have.

THE CALL BOUNDARY
-----------------
A call is a maximal run of MODEL-produced items. Everything else — tool results, and
user messages that arrive mid-trajectory — is environment input that closes the current
call and starts the next one. This is more general than "cut after every tool output":
it also handles the 1.5k mid-trajectory user turns (a human replying in the thread).

    input of call k  = items[:start_k]  (+ the tool schemas, sent on every request)
    output of call k = items[start_k : start_k + len(run)]

CACHING
-------
Call k's input begins with call k-1's input, so that prefix is cache-eligible; the model
output and tool result appended since then are not. Cached share is therefore
items[:start_{k-1}] + tools, billed at the cache-READ rate. This is a LOWER bound on the
cache credit — a provider that also caches the previous response would credit more. There
is no `usage.cached_tokens` in the export to check against, so this is an estimate, like
every token count here (serialized chars / 4).

Usage:  python scripts/decompose.py export/trajectories_v1_01.jsonl
Writes: results/calls.jsonl — one record per reconstructed call, joinable to
        results/trajectory_index.jsonl on `row`.
Importable: decompose, iter_calls.
"""
import json, sys, itertools, collections, hashlib
from pathlib import Path
from cost_model import DEFAULT_PRICING, load_pricing, price_of

# Items the MODEL produces. Anything else came from the environment.
MODEL_ITEM_TYPES = {"reasoning", "function_call", "custom_tool_call"}


def is_model_item(item):
    t = item.get("type")
    if t in MODEL_ITEM_TYPES:
        return True
    # message items carry the role; only the assistant's are model output.
    # (type is absent on some legacy message items — the role still decides.)
    if t == "message" or t is None:
        return item.get("role") == "assistant"
    return False


def decompose(items):
    """Split a trajectory's item list into the calls that produced it.

    Returns [(start, stop)] index pairs: call k had input items[:start] and produced
    items[start:stop]. Leading system/user items form the first call's input, so the
    first span starts after them.
    """
    calls, i = [], 0
    while i < len(items):
        if not is_model_item(items[i]):      # environment input — accumulates into the
            i += 1                            # next call's prompt
            continue
        start = i
        while i < len(items) and is_model_item(items[i]):
            i += 1
        calls.append((start, i))
    return calls


def iter_calls(row, row_idx, pricing):
    """Yield one priced record per call in this trajectory."""
    items = row["input"]
    model = row["model"]
    pu, pc, po = price_of(model, pricing)

    tok = [len(json.dumps(it)) // 4 for it in items]
    prefix = list(itertools.accumulate(tok, initial=0))   # prefix[k] = tokens of items[:k]
    tools = row.get("tools") or []
    tools_tok = len(json.dumps(tools)) // 4               # resent on every request

    spans = decompose(items)
    traj_id = hashlib.sha1(json.dumps(items, sort_keys=True).encode()).hexdigest()[:12]
    prev_start = None

    for k, (start, stop) in enumerate(spans):
        inp = prefix[start] + tools_tok
        cached = min(prefix[prev_start] + tools_tok, inp) if prev_start is not None else 0
        uncached = inp - cached
        out_tok = prefix[stop] - prefix[start]
        cost = (uncached * pu + cached * pc + out_tok * po) / 1e6
        prev_start = start

        produced = items[start:stop]
        yield {
            "row": row_idx,
            "trajectory_id": traj_id,
            "call_index": k,
            "n_calls": len(spans),
            "is_final": k == len(spans) - 1,
            "model": model,
            "family": "claude" if model.startswith("claude") else "gpt",
            "input_tokens_est": inp,
            "cached_tokens_est": cached,
            "uncached_tokens_est": uncached,
            "output_tokens_est": out_tok,
            "tools_tokens_est": tools_tok,
            "cost_usd_est": round(cost, 8),
            "output_kinds": [it.get("type") or it.get("role") for it in produced],
            "tool_names": [it["name"] for it in produced if it.get("name")],
            "n_tools_available": len(tools),
        }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "export/trajectories_v1_01.jsonl"
    pricing = load_pricing()
    rows = [json.loads(l) for l in open(src) if l.strip()]

    Path("results").mkdir(exist_ok=True)
    dest = Path("results") / "calls.jsonl"

    n_calls = 0
    total = naive = out_usd = 0.0
    tot_in = tot_out = tot_unc = 0
    by_model = collections.defaultdict(float)
    by_index = collections.defaultdict(float)
    per_traj = []

    with open(dest, "w") as f:
        for i, row in enumerate(rows):
            traj = 0.0
            for rec in iter_calls(row, i, pricing):
                f.write(json.dumps(rec) + "\n")
                n_calls += 1
                c = rec["cost_usd_est"]
                total += c
                traj += c
                out_usd += rec["output_tokens_est"] * price_of(rec["model"], pricing)[2] / 1e6
                tot_in += rec["input_tokens_est"]
                tot_out += rec["output_tokens_est"]
                tot_unc += rec["uncached_tokens_est"]
                by_model[rec["model"]] += c
                by_index[min(rec["call_index"], 20)] += c
            per_traj.append(traj)
            # what the starter kit sees: the row as a single input-only call
            pu = price_of(row["model"], pricing)[0]
            naive += ((len(json.dumps(row["input"])) // 4) + (len(json.dumps(row.get("tools") or [])) // 4)) * pu / 1e6

    calls_per = collections.Counter()
    print(f"rows={len(rows)}  reconstructed calls={n_calls:,}  "
          f"(mean {n_calls/len(rows):.1f} per trajectory)")
    print(f"est. input tokens billed: {tot_in:,}  uncached: {tot_unc:,} ({tot_unc/tot_in:.1%})")
    print(f"est. output tokens RECOVERED: {tot_out:,}  (the export has no `usage` — these are estimates)")
    print()
    print(f"cost, cache-aware, input+output : ${total:,.2f}")
    print(f"cost as the starter kit sees it : ${naive:,.2f}  -> {total/naive:.2f}x undercount")
    print(f"  of which output-side          : ${out_usd:,.2f} ({out_usd/total:.1%})")
    print()
    print("spend by model:")
    for m, v in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {m:20} ${v:8.2f}  ({v/total:5.1%})")
    print()
    first = by_index[0] / total
    print(f"spend in the first call of each trajectory: {first:.1%}  "
          f"(uncached cold start — the biggest single line item)")
    per_traj.sort(reverse=True)
    top10 = sum(per_traj[:len(per_traj)//10]) / total
    print(f"spend in the top 10% of trajectories:       {top10:.1%}")
    print()
    print(f"wrote {dest}")
    print("NOTE: all token counts are estimates (serialized chars / 4), not measured.")


if __name__ == "__main__":
    main()
