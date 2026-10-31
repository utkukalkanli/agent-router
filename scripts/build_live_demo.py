#!/usr/bin/env python3
"""Build the aggregate-only static Routemaxxing demo (deployed to Vercel).

This exporter intentionally whitelists aggregate evaluation fields. It never reads or
ships prompts, tools, trajectory ids, route ledgers, or any other challenge row data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "routemaxxing"
OUTPUT = ROOT / "docs" / "data.js"

PRICING = {
    "claude-sonnet-5": [2.00, 0.20, 10.00],
    "claude-opus-5": [5.00, 0.50, 25.00],
    "claude-fable-5": [10.00, 1.00, 50.00],
    "gpt-5.6-luna": [0.20, 0.02, 1.20],
    "gpt-5.6-terra": [2.00, 0.20, 12.00],
    "gpt-5.6-sol": [5.00, 0.50, 30.00],
}

CERTIFIED_DOWNGRADES = {
    "claude-fable-5": "claude-sonnet-5",
    "claude-opus-5": "claude-sonnet-5",
    "gpt-5.6-sol": "gpt-5.6-terra",
}

COMPARISON_POLICIES = {
    "logged_behavior",
    "corrected_heuristic_baseline",
    "current_routemaxxing_certified",
    "final_improved_certified",
    "adaptive_conditional",
}


def _number(value: str):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def build() -> dict:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    output = json.loads(
        (RESULTS / "output_length_sensitivity.json").read_text(encoding="utf-8")
    )
    with (RESULTS / "frontier.csv").open(encoding="utf-8") as handle:
        frontier = [
            {
                key: _number(value)
                for key, value in row.items()
                if key in {
                    "adopt_frac", "cost_usd_est",
                    "behavior_adjusted_cost_usd_est", "savings_pct",
                    "failure_delta", "failure_upper95", "coverage",
                    "effective_n", "selected",
                }
            }
            for row in csv.DictReader(handle)
        ]
    with (RESULTS / "ablation_comparison.csv").open(encoding="utf-8") as handle:
        comparison = [
            {
                key: _number(value)
                for key, value in row.items()
                if key in {
                    "policy", "cost_usd_est", "behavior_adjusted_cost_usd_est",
                    "savings_pct", "behavior_adjusted_savings_pct",
                    "failure_upper95", "coverage", "effective_n", "note",
                }
            }
            for row in csv.DictReader(handle)
            if row["policy"] in COMPARISON_POLICIES
        ]
    selected = summary["validation_selected_point"]
    current = summary["current_validation_point"]
    return {
        "meta": {
            "name": "Routemaxxing Router Lab",
            "split": "public validation",
            "token_estimator": "serialized JSON characters / 4; not measured usage",
            "held_out": "not used for development",
            "data_policy": "aggregate evidence only; no challenge trajectories included",
        },
        "headline": {
            "logged_cost": 88.8784,
            "current_cost": current["cost_usd_est"],
            "routed_cost": selected["cost_usd_est"],
            "behavior_adjusted_cost": selected["behavior_adjusted_cost_usd_est"],
            "savings_pct": selected["savings_pct"],
            "failure_upper95_pp": selected["failure_upper95"] * 100,
            "coverage_pct": selected["coverage"] * 100,
            "effective_n": selected["effective_n"],
            "minimum_propensity": selected["min_target_propensity"],
        },
        "pricing": PRICING,
        "certified_downgrades": CERTIFIED_DOWNGRADES,
        "families": {
            "claude": ["claude-sonnet-5", "claude-opus-5", "claude-fable-5"],
            "gpt": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
        },
        "support": {
            "minimum_train_n": 20,
            "gpt-5.6-luna": {"train_n": 12, "eligible": False},
        },
        "output_length_estimates": output["model_expected_output_tokens_est"],
        "frontier": frontier,
        "comparison": comparison,
        "constraints": [
            "One model for the complete certified trajectory",
            "Never upgrade above the configured default",
            "Preserve the tool-dialect model family",
            "Require positive fixed and behavior-adjusted savings",
            "Unsupported actions fall back to the default",
            "Primary proxy upper95 must remain at or below +2 percentage points",
        ],
        "known_weakness": (
            "Proxy failure is not measured task quality. Outcome regression can fail "
            "under residual confounding, and adaptive state transitions are conditional "
            "on the logged trajectory rather than identified end to end."
        ),
    }


def main() -> None:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "window.ROUTEMAXXING_DEMO = "
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    print("wrote aggregate-only demo data to %s" % OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
