#!/usr/bin/env python3
"""Compare every router version in results/*/ against the baseline (results/routes.jsonl).

Never writes to results/routes.jsonl, results/frontier.csv (the baseline's own files) —
only reads them. Everything it writes is new: results/<version>/frontier.csv for your
own versions, results/comparison.csv, results/comparison.png, results/experiments.csv.

Usage: python scripts/compare.py
"""
import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_frontier import frontier_rows  # noqa: E402

EXPERIMENT_FIELDS = ["version", "description", "cost_usd_full_adopt",
                      "quality_full_adopt", "cost_delta_vs_baseline_pct"]


def discover_versions(results_root):
    """{"baseline": results/routes.jsonl, "<name>": results/<name>/routes.jsonl, ...}
    for every subfolder of results/ that contains a routes.jsonl."""
    results_root = Path(results_root)
    versions = {}
    baseline = results_root / "routes.jsonl"
    if baseline.exists():
        versions["baseline"] = baseline
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and (d / "routes.jsonl").exists():
            versions[d.name] = d / "routes.jsonl"
    return versions


def summarize(routes_path):
    """cost + quality at full adoption (adopt_frac == 1.0) of a version's routes.jsonl."""
    recs = [json.loads(l) for l in open(routes_path) if l.strip()]
    full = frontier_rows(recs)[-1]
    return {"cost_usd_full_adopt": full["cost_usd"], "quality_full_adopt": full["quality_placeholder"]}


def _delta_pct(cost, baseline_cost):
    if baseline_cost is None or baseline_cost == 0:
        return None
    return round((cost / baseline_cost - 1) * 100, 2)


def write_comparison_csv(results_root, summary):
    baseline_cost = summary.get("baseline", {}).get("cost_usd_full_adopt")
    out = Path(results_root) / "comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["version", "cost_usd_full_adopt", "quality_full_adopt",
                                           "cost_delta_vs_baseline_pct"])
        w.writeheader()
        for name, s in summary.items():
            w.writerow({"version": name, **s,
                        "cost_delta_vs_baseline_pct": _delta_pct(s["cost_usd_full_adopt"], baseline_cost)})
    return out


def upsert_experiments_csv(results_root, summary):
    """One row per version EXCLUDING baseline (it's the reference, not your experiment).
    Preserves a hand-written `description` column across re-runs."""
    baseline_cost = summary.get("baseline", {}).get("cost_usd_full_adopt")
    exp_path = Path(results_root) / "experiments.csv"
    existing = {}
    if exp_path.exists():
        with open(exp_path, newline="") as f:
            existing = {row["version"]: row for row in csv.DictReader(f)}
    with open(exp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        w.writeheader()
        for name, s in summary.items():
            if name == "baseline":
                continue
            w.writerow({
                "version": name,
                "description": existing.get(name, {}).get("description", ""),
                "cost_usd_full_adopt": s["cost_usd_full_adopt"],
                "quality_full_adopt": s["quality_full_adopt"],
                "cost_delta_vs_baseline_pct": _delta_pct(s["cost_usd_full_adopt"], baseline_cost),
            })
    return exp_path


def main(results_root=None):
    results_root = Path(results_root) if results_root else (SCRIPTS_DIR.parent / "results")
    versions = discover_versions(results_root)

    summary = {}
    for name, routes_path in versions.items():
        try:
            recs = [json.loads(l) for l in open(routes_path) if l.strip()]
            rows = frontier_rows(recs)
        except (json.JSONDecodeError, KeyError, IndexError, ZeroDivisionError) as e:
            print(f"skip {name}: could not parse {routes_path} ({e})")
            continue
        full = rows[-1]
        summary[name] = {"cost_usd_full_adopt": full["cost_usd"], "quality_full_adopt": full["quality_placeholder"]}
        if name != "baseline":  # never rewrite the baseline's own frontier.csv
            out_csv = routes_path.parent / "frontier.csv"
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    if not summary:
        print("no versions found under", results_root)
        return

    comp_path = write_comparison_csv(results_root, summary)
    print(f"wrote {comp_path}")
    for name, s in summary.items():
        print(f"  {name:20s} cost=${s['cost_usd_full_adopt']:.4f}  quality={s['quality_full_adopt']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for name, routes_path in versions.items():
            if name not in summary:
                continue
            recs = [json.loads(l) for l in open(routes_path) if l.strip()]
            rows = frontier_rows(recs)
            style = "--" if name == "baseline" else "-o"
            plt.plot([r["cost_usd"] for r in rows], [r["quality_placeholder"] for r in rows], style, label=name)
        plt.xlabel("cost (USD, estimated tokens, cache-aware)"); plt.ylabel("quality (placeholder)")
        plt.title("Cost–quality frontier — all versions"); plt.legend(); plt.tight_layout()
        png = Path(results_root) / "comparison.png"
        plt.savefig(png, dpi=150)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed — skipped comparison.png")

    exp_path = upsert_experiments_csv(results_root, summary)
    print(f"wrote {exp_path} (edit its 'description' column by hand to note what each version tried)")


if __name__ == "__main__":
    main()
