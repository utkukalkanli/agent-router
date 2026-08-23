import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cost_model import load_pricing
from routemaxxing import (
    _value_threshold,
    Call,
    Outcome,
    Trajectory,
    choose_initial,
    choose_cost_effective_initial,
    corrected_baseline,
    infer_family,
    make_routes,
    outcome_from_items,
    route_cost,
    route_cost_at_decision,
    trajectory_from_row,
)


def message(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def tool(name):
    return {"type": "function", "name": name, "parameters": {}}


def complete_row(model="claude-opus-5"):
    items = [
        message("system", "agent"), message("user", "fix the tests"),
        {"type": "function_call", "name": "bash", "arguments": '{"command":"test"}'},
        {"type": "function_call_output", "output": "ok"},
        message("assistant", "Done."),
    ]
    return {"model": model, "input": items, "tools": [tool("bash"), tool("file_edit")]}


class DummyModel:
    def __init__(self, risks, context_p90=10**9):
        self.candidates = list(risks)
        self.risks = risks
        self.context_p90 = context_p90

    def risk(self, trajectory, model):
        return self.risks[model]

    def propensity(self, trajectory):
        return {model: 1 / len(self.candidates) for model in self.candidates}


class RoutemaxxingTest(unittest.TestCase):
    def test_family_is_inferred_from_tool_dialect(self):
        self.assertEqual(infer_family([tool("bash"), tool("file_edit")]), "claude")
        self.assertEqual(infer_family([tool("shell_command"), tool("apply_patch")]), "gpt")

    def test_complete_history_decomposes_and_final_output_is_separately_observed(self):
        trajectory = trajectory_from_row(complete_row(), 0)
        self.assertEqual(trajectory.trajectory_id, __import__("hashlib").sha1(
            json.dumps(complete_row()["input"], sort_keys=True).encode()
        ).hexdigest()[:12])
        self.assertEqual(len(trajectory.calls), 2)
        self.assertFalse(trajectory.calls[0].is_final)
        self.assertTrue(trajectory.calls[1].is_final)
        self.assertGreater(trajectory.calls[1].output_tokens, 0)

    def test_failure_proxy_keeps_terminal_and_official_components_separate(self):
        items = complete_row()["input"][:-1] + [message("assistant", "Sorry, I was unable to finish.")]
        outcome = outcome_from_items(items)
        self.assertEqual(outcome.observed_failure, 1)
        self.assertEqual(outcome.official_failure, 0)

    def test_certified_initial_choice_never_upgrades_default(self):
        trajectory = trajectory_from_row(complete_row("claude-opus-5"), 0)
        model = DummyModel({
            "claude-sonnet-5": 0.30,
            "claude-opus-5": 0.20,
            "claude-fable-5": 0.10,
        })
        chosen, _, _ = choose_initial(trajectory, model, load_pricing(), margin=0.0)
        self.assertEqual(chosen, "claude-opus-5")

    def test_adaptive_route_switches_at_most_once_and_not_above_default(self):
        row = complete_row("claude-opus-5")
        row["input"] = row["input"][:-1] + [
            {"type": "function_call", "name": "bash", "arguments": '{"command":"test"}'},
            {"type": "function_call_output", "output": "ERROR: failed"},
            message("assistant", "Done."),
        ]
        trajectory = trajectory_from_row(row, 0)
        model = DummyModel({
            "claude-sonnet-5": 0.11,
            "claude-opus-5": 0.10,
            "claude-fable-5": 0.09,
        })
        result = make_routes(trajectory, model, load_pricing())
        route = result["adaptive"]
        switches = sum(route[i] != route[i - 1] for i in range(1, len(route)))
        self.assertLessEqual(switches, 1)
        self.assertNotIn("claude-fable-5", route)
        if switches:
            switch_index = next(i for i in range(1, len(route)) if route[i] != route[i - 1])
            self.assertIn("cache-break-even", result["adaptive_reasons"][switch_index])

    def test_model_switch_resets_cache_and_has_positive_penalty(self):
        trajectory = trajectory_from_row(complete_row("claude-opus-5"), 0)
        pricing = load_pricing()
        same_cost, same_penalty = route_cost(
            trajectory, ["claude-sonnet-5"] * len(trajectory.calls), pricing,
        )
        switched_cost, switch_penalty = route_cost(
            trajectory, ["claude-sonnet-5", "claude-opus-5"], pricing,
        )
        self.assertEqual(same_penalty, 0)
        self.assertGreater(switch_penalty, 0)
        self.assertGreater(switched_cost, same_cost)

    def test_adaptive_break_even_does_not_see_current_future_output(self):
        trajectory = trajectory_from_row(complete_row("claude-opus-5"), 0)
        route = ["claude-sonnet-5", "claude-opus-5"]
        before = route_cost_at_decision(trajectory, route, load_pricing())
        trajectory.calls[-1].output_tokens += 1_000_000
        after = route_cost_at_decision(trajectory, route, load_pricing())
        self.assertEqual(before, after)

    def test_cost_effective_choice_falls_back_for_unsupported_default(self):
        row = complete_row("gpt-5.6-luna")
        row["tools"] = [tool("shell_command"), tool("apply_patch")]
        trajectory = trajectory_from_row(row, 0)
        model = DummyModel({"gpt-5.6-terra": 0.10, "gpt-5.6-sol": 0.08})
        chosen, reasons, _, proposal = choose_cost_effective_initial(
            trajectory, model, load_pricing(), {}, None,
        )
        self.assertEqual(chosen, "gpt-5.6-luna")
        self.assertIn("unsupported-default-fallback", reasons)
        self.assertEqual(proposal["fixed_savings"], 0.0)

    def test_cost_effective_choice_is_same_family_cheaper_and_positive(self):
        trajectory = trajectory_from_row(complete_row("claude-fable-5"), 0)
        model = DummyModel({
            "claude-sonnet-5": 0.11,
            "claude-opus-5": 0.10,
            "claude-fable-5": 0.09,
        })
        chosen, _, _, proposal = choose_cost_effective_initial(
            trajectory, model, load_pricing(), {}, None,
        )
        self.assertIn(chosen, {"claude-sonnet-5", "claude-opus-5"})
        self.assertLess(
            load_pricing()[chosen][0], load_pricing()[trajectory.logged_model][0],
        )
        self.assertGreater(proposal["fixed_savings"], 0.0)
        self.assertGreater(proposal["adjusted_savings"], 0.0)

    def test_opening_features_do_not_change_with_future_items(self):
        first = trajectory_from_row(complete_row(), 0)
        row = complete_row()
        row["input"].insert(-1, {"type": "function_call_output", "output": "future error"})
        second = trajectory_from_row(row, 1)
        self.assertEqual(first.feature_values, second.feature_values)

    def test_corrected_baseline_does_not_use_eventual_depth(self):
        first = trajectory_from_row(complete_row("claude-opus-5"), 0)
        second = trajectory_from_row(complete_row("claude-opus-5"), 1)
        second.calls.extend([second.calls[-1]] * 100)
        model = DummyModel({
            "claude-sonnet-5": 0.1,
            "claude-opus-5": 0.1,
            "claude-fable-5": 0.1,
        })
        self.assertEqual(
            corrected_baseline(first, model, load_pricing()),
            corrected_baseline(second, model, load_pricing())[:len(first.calls)],
        )

    def test_training_value_threshold_applies_to_proposal_scores(self):
        trajectories = [trajectory_from_row(complete_row(), i) for i in range(4)]
        for i, trajectory in enumerate(trajectories):
            trajectory.trajectory_id = str(i)
        proposals = {
            str(i): {
                "certified": ["claude-sonnet-5"],
                "proposal": {"value_per_risk": float(i)},
            } for i in range(4)
        }
        self.assertEqual(_value_threshold(trajectories, proposals, 0.5), 2.0)


if __name__ == "__main__":
    unittest.main()
