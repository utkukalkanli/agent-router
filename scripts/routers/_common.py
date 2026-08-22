#!/usr/bin/env python3
"""Shared I/O for router variants in scripts/routers/.

Each version file defines its own `route_trajectory(calls) -> [model_id, ...]`
policy and calls `write_routes(route_trajectory)` — never touches
`scripts/baseline_router.py` or `results/routes.jsonl` directly.

Usage inside a version file (e.g. scripts/routers/v1_myrouter.py):

    from _common import write_routes

    def route_trajectory(calls):
        ...
        return [...]

    if __name__ == "__main__":
        write_routes(route_trajectory)
"""
import sys, json
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # scripts/
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from load_trajectories import iter_requests, group_trajectories, est_tokens  # noqa: E402
from cost_model import trajectory_cost, logged_route, load_pricing  # noqa: E402


def write_routes(route_trajectory, export="export", name=None, results_root=None):
    """Run route_trajectory(calls) over the dataset and write
    results/<name>/routes.jsonl in the same schema as the baseline router.

    `name` defaults to the calling script's filename stem (v1_heuristic.py ->
    "v1_heuristic"). Returns (tot_logged_usd, tot_routed_usd).
    """
    name = name or Path(sys.argv[0]).stem
    results_root = Path(results_root) if results_root else (SCRIPTS_DIR.parent / "results")

    pricing = load_pricing()
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    outdir = results_root / name
    outdir.mkdir(parents=True, exist_ok=True)

    tot_logged = tot_routed = 0.0
    with open(outdir / "routes.jsonl", "w") as out:
        for key, calls in groups.items():
            logged = logged_route(calls)
            routed = route_trajectory(calls)
            c_logged, _ = trajectory_cost(calls, logged, pricing)
            c_routed, _ = trajectory_cost(calls, routed, pricing)
            tot_logged += c_logged
            tot_routed += c_routed
            out.write(json.dumps({
                "trajectory": key, "n_calls": len(calls), "logged_model": logged[0],
                "route": routed, "cost_logged_usd": round(c_logged, 6),
                "cost_routed_usd": round(c_routed, 6),
                "switches": sum(1 for i in range(1, len(routed)) if routed[i] != routed[i - 1]),
            }) + "\n")

    print(f"[{name}] logged cost: ${tot_logged:,.4f}  routed cost: ${tot_routed:,.4f}"
          f"  ({(tot_routed / tot_logged - 1):+.1%})" if tot_logged else f"[{name}] no trajectories found")
    print(f"wrote {outdir / 'routes.jsonl'}")
    return tot_logged, tot_routed
