import copy
import json
import unittest

from v2_oracle import (
    Prediction,
    _fold,
    build_trajectory,
    choose_quality_constrained,
    context_from_row,
    cost_for_model,
    fit_router,
    outcome_score,
    tool_result_failed,
)


SYS = {"role": "system", "content": "You are Viktor."}
USER = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Please inspect the report and send a summary."}],
}
TOOLS = [{"type": "function", "name": "bash", "description": "Run a command"}]


def row_with_result(output="ok", model="claude-opus-5"):
    return {
        "model": model,
        "tools": TOOLS,
        "input": [
            SYS,
            USER,
            {"type": "function_call", "name": "bash", "call_id": "c1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": output},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ],
    }


def prediction(model, cost, quality, se=0.01, effective=20.0):
    return Prediction(
        model=model,
        uncached_tokens=100.0,
        cached_tokens=200.0,
        output_tokens=20.0,
        cost_usd=cost,
        quality=quality,
        quality_se=se,
        effective_neighbors=effective,
        nearest_distance=0.1,
    )


class OracleRouterTests(unittest.TestCase):
    def test_context_features_stop_before_first_model_output(self):
        row = row_with_result("ok")
        changed = copy.deepcopy(row)
        changed["input"][3]["output"] = "a completely different and very long result " * 100
        changed["input"][4]["content"][0]["text"] = "A different final answer"

        before = context_from_row(row)
        after = context_from_row(changed)

        self.assertEqual(before, after)

    def test_build_trajectory_recovers_calls_output_and_cache(self):
        record = build_trajectory(row_with_result(), 0)

        self.assertEqual(record.n_calls, 2)
        self.assertGreater(record.context.initial_tokens, 0)
        self.assertGreater(record.cached_tokens, 0)
        self.assertGreater(record.output_tokens, 0)
        self.assertGreater(record.observed_cost, 0)

    def test_cost_for_model_uses_input_cache_and_output_rates(self):
        record = build_trajectory(row_with_result(), 0)
        pricing = {
            "cheap": [1.0, 0.1, 2.0],
            "expensive": [2.0, 0.2, 4.0],
            "_default": [1.0, 0.1, 2.0],
        }

        cheap = cost_for_model(record, "cheap", pricing)
        expensive = cost_for_model(record, "expensive", pricing)

        self.assertAlmostEqual(expensive, cheap * 2)

    def test_failure_detector_does_not_match_documentation_that_mentions_errors(self):
        successful_doc = json.dumps({
            "content": "Exit code: 0\nDocumentation: handle errors and failed retries safely.",
            "exit_code": 0,
        })
        failed_command = json.dumps({"content": "command timed out", "exit_code": 124})

        self.assertFalse(tool_result_failed(successful_doc))
        self.assertTrue(tool_result_failed(failed_command))

    def test_outcome_score_treats_a_successful_retry_as_recovered(self):
        row = row_with_result(json.dumps({"content": "bad", "exit_code": 1}))
        row["input"][4:4] = [
            {"type": "function_call", "name": "bash", "call_id": "c2", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c2", "output": "ok"},
        ]

        recovered = outcome_score(row)
        unresolved = outcome_score(row_with_result(json.dumps({"content": "bad", "exit_code": 1})))

        self.assertGreater(recovered, unresolved)
        self.assertLess(recovered, 1.0)

    def test_quality_constraint_chooses_cheap_near_tie(self):
        predictions = {
            "cheap": prediction("cheap", 1.0, 0.985),
            "strong": prediction("strong", 4.0, 0.99),
        }

        action = choose_quality_constrained(predictions, tolerance=0.01)

        self.assertEqual(action, "cheap")

    def test_quality_constraint_rejects_uncertain_cheap_model(self):
        predictions = {
            "cheap": prediction("cheap", 1.0, 0.99, se=0.05),
            "strong": prediction("strong", 4.0, 0.99, se=0.005),
        }

        action = choose_quality_constrained(predictions, tolerance=0.01, uncertainty_z=1.0)

        self.assertEqual(action, "strong")

    def test_fitted_router_accepts_a_live_first_call_request(self):
        training = []
        for model in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5"):
            for index in range(3):
                row = copy.deepcopy(row_with_result(f"ok {index}", model=model))
                row["input"][1]["content"][0]["text"] += f" example {model} {index}"
                training.append(row)
        router = fit_router(training, version="router_v4_conservative")
        live = copy.deepcopy(row_with_result())
        live["input"] = live["input"][:2]

        decision = router.route(live)

        self.assertIn(decision["model"], {"claude-sonnet-5", "claude-opus-5", "claude-fable-5"})
        self.assertEqual(set(decision["estimates"]), {"claude-sonnet-5", "claude-opus-5", "claude-fable-5"})
        self.assertTrue(all(value["cost_usd"] > 0 for value in decision["estimates"].values()))
        self.assertTrue(all(value["total_tokens_p50"] > 0 for value in decision["estimates"].values()))

    def test_normalized_task_key_groups_redacted_recurring_jobs(self):
        first = copy.deepcopy(row_with_result())
        second = copy.deepcopy(row_with_result())
        first["input"][1]["content"][0]["text"] = (
            "<system>- Task (from scheduler): Monitor PII_PROJECT_1 every 10 minutes "
            "and notify PII_PERSON_4.\n- Cron path: `crons/PII_PROJECT_1/watch/`</system>"
        )
        second["input"][1]["content"][0]["text"] = (
            "<system>- Task (from scheduler): Monitor PII_PROJECT_9 every 20 minutes "
            "and notify PII_PERSON_8.\n- Cron path: `crons/PII_PROJECT_9/watch/`</system>"
        )

        left = context_from_row(first)
        right = context_from_row(second)

        self.assertEqual(left.task_key, right.task_key)
        self.assertEqual(_fold(left, 5, group_tasks=True), _fold(right, 5, group_tasks=True))

    def test_recurring_task_estimates_report_analogue_support(self):
        training = [
            copy.deepcopy(row_with_result(f"ok {index}", model=model))
            for model in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5")
            for index in range(2)
        ]
        router = fit_router(training, version="router_v6_analogue")
        live = copy.deepcopy(row_with_result())
        live["input"] = live["input"][:2]

        estimates = router.estimate(live)

        self.assertTrue(all(value["analogue_neighbors"] == len(training) for value in estimates.values()))


if __name__ == "__main__":
    unittest.main()
