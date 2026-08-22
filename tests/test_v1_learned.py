from v1_learned import (outcome_score, bucket_key, fit_bucket_routes, compute_edges,
                         _tercile_edges, make_route_trajectory)

SYS = {"role": "system", "content": "You are Viktor."}
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do task"}]}
ASSISTANT_MSG = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done!"}]}
HEDGE_ASSISTANT_MSG = {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "I'm sorry, I was unable to finish this."}]}


def fco(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def fc(call_id, name="run_shell"):
    return {"type": "function_call", "name": name, "call_id": call_id}


# ---- outcome_score ----

def test_outcome_score_clean_completion_no_errors():
    calls = [{"model": "claude-opus-5", "tools": [],
              "input": [SYS, USER, fc("1"), fco("1", "ok, done"), ASSISTANT_MSG]}]
    assert outcome_score(calls) == 1.0


def test_outcome_score_penalizes_function_call_errors():
    calls = [{"model": "claude-opus-5", "tools": [],
              "input": [SYS, USER, fc("1"), fco("1", "Error: failed"), ASSISTANT_MSG]}]
    assert outcome_score(calls) == 0.6


def test_outcome_score_penalizes_unfinished_trajectory():
    calls = [{"model": "claude-opus-5", "tools": [],
              "input": [SYS, USER, fc("1")]}]  # ends mid function_call, no output at all
    assert outcome_score(calls) == 0.4


def test_outcome_score_catches_broader_failure_language_than_literal_error():
    calls = [{"model": "claude-opus-5", "tools": [],
              "input": [SYS, USER, fc("1"), fco("1", "Request timeout, please retry"), ASSISTANT_MSG]}]
    assert outcome_score(calls) == 0.6  # "timeout" never says "error" but is still a failure signal


def test_outcome_score_penalizes_hedging_final_message():
    calls = [{"model": "claude-opus-5", "tools": [],
              "input": [SYS, USER, fc("1"), fco("1", "ok"), HEDGE_ASSISTANT_MSG]}]
    assert outcome_score(calls) == 0.7  # ended with a message, but it apologizes/admits failure


# ---- quantile edges ----

def test_tercile_edges_splits_sorted_values_into_thirds():
    assert _tercile_edges([10, 20, 30, 40, 50, 60]) == (30, 50)


def test_tercile_edges_handles_single_value():
    assert _tercile_edges([5]) == (5, 5)


def test_compute_edges_derives_tool_edges_from_the_dataset_itself():
    groups = {
        "g0": [{"model": "m", "tools": [{"type": "function", "name": f"t{i}"} for i in range(0)],
                "input": [SYS, USER]}],
        "g1": [{"model": "m", "tools": [{"type": "function", "name": f"t{i}"} for i in range(10)],
                "input": [SYS, USER]}],
        "g2": [{"model": "m", "tools": [{"type": "function", "name": f"t{i}"} for i in range(20)],
                "input": [SYS, USER]}],
    }
    assert compute_edges(groups)["tools"] == (10, 20)


# ---- bucket_key ----

EDGES = {"tokens": (5_000, 20_000), "tools": (3, 8), "calls": (2, 5)}


def test_bucket_key_small_task():
    calls = [{"model": "claude-opus-5", "tools": [{"type": "function", "name": "t1"}],
              "input": [SYS, USER]}]
    assert bucket_key(calls, EDGES) == ("small", "few", "short")


def test_bucket_key_large_task():
    big_text = "filler " * 20_000  # pushes est_tokens well past the "large" edge
    calls = [{"model": "claude-opus-5", "tools": [{"type": "function", "name": f"t{i}"} for i in range(20)],
              "input": [SYS, {"type": "message", "role": "user",
                               "content": [{"type": "input_text", "text": big_text}]}]}
             for _ in range(6)]
    assert bucket_key(calls, EDGES) == ("large", "many", "long")


# ---- fit_bucket_routes ----

CLEAN_INPUT = [SYS, USER, fc("1"), fco("1", "ok"), ASSISTANT_MSG]
ERROR_INPUT = [SYS, USER, fc("1"), fco("1", "Error: failed"), ASSISTANT_MSG]
# 5 outputs, 1 error -> outcome_score 0.92: clears an old 0.9 relative floor but not a 0.98 one
SLIGHTLY_FLAWED_INPUT = [SYS, USER] + [x for i in range(1, 6) for x in (
    fc(str(i)), fco(str(i), "Error: failed" if i == 5 else "ok"))] + [ASSISTANT_MSG]

PRICING = {"claude-opus-5": [5.0, 0.5, 25.0], "claude-sonnet-5": [2.0, 0.2, 10.0], "_default": [2.0, 0.2, 10.0]}


def one_call_group(model, input_items):
    return [{"model": model, "tools": [], "input": input_items}]


def repeat_groups(model, input_items, n, prefix):
    """n separate one-call trajectories for `model`, all landing in the same bucket."""
    return {f"{prefix}{i}": one_call_group(model, input_items) for i in range(n)}


def test_fit_bucket_routes_picks_cheaper_model_when_it_meets_relative_threshold():
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),
              **repeat_groups("claude-sonnet-5", CLEAN_INPUT, 3, "b")}
    routes = fit_bucket_routes(groups, PRICING, threshold_type="relative", threshold_value=0.9, edges=EDGES)
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-sonnet-5"


def test_fit_bucket_routes_falls_back_to_only_observed_model():
    groups = repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a")
    routes = fit_bucket_routes(groups, PRICING, threshold_type="relative", threshold_value=0.9, edges=EDGES)
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-opus-5"


def test_fit_bucket_routes_relative_mode_rejects_cheap_model_with_low_score():
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),   # score 1.0
              **repeat_groups("claude-sonnet-5", ERROR_INPUT, 3, "b")}  # score 0.6, below 0.9*1.0
    routes = fit_bucket_routes(groups, PRICING, threshold_type="relative", threshold_value=0.9, edges=EDGES)
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-opus-5"


def test_fit_bucket_routes_default_threshold_rejects_a_merely_ok_cheap_model():
    # real data showed outcome_score barely varies across models (0.85-1.0 band) --
    # a lenient 0.9 relative floor let the cheapest model win almost every bucket.
    # The default must be strict enough to reject a model that isn't near-tied with the best.
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),            # score 1.0
              **repeat_groups("claude-sonnet-5", SLIGHTLY_FLAWED_INPUT, 3, "b")}  # score 0.92
    routes = fit_bucket_routes(groups, PRICING, edges=EDGES)  # no threshold_value -> use the default
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-opus-5"


def test_fit_bucket_routes_absolute_mode_accepts_same_cheap_model_with_lower_bar():
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),
              **repeat_groups("claude-sonnet-5", ERROR_INPUT, 3, "b")}
    routes = fit_bucket_routes(groups, PRICING, threshold_type="absolute", threshold_value=0.5, edges=EDGES)
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-sonnet-5"


def test_fit_bucket_routes_falls_back_to_best_when_nothing_meets_absolute_threshold():
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),
              **repeat_groups("claude-sonnet-5", ERROR_INPUT, 3, "b")}
    routes = fit_bucket_routes(groups, PRICING, threshold_type="absolute", threshold_value=1.5, edges=EDGES)
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-opus-5"


def test_fit_bucket_routes_ignores_candidate_with_too_few_samples():
    # cheap model scores perfectly but was only seen once -> not enough evidence to trust it
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),
              **repeat_groups("claude-sonnet-5", CLEAN_INPUT, 1, "b")}
    routes = fit_bucket_routes(groups, PRICING, threshold_type="relative", threshold_value=0.9,
                                min_samples=3, edges=EDGES)
    assert routes[bucket_key(one_call_group("x", CLEAN_INPUT), EDGES)] == "claude-opus-5"


def test_fit_bucket_routes_skips_bucket_when_no_model_has_enough_samples():
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 2, "a"),
              **repeat_groups("claude-sonnet-5", CLEAN_INPUT, 2, "b")}
    routes = fit_bucket_routes(groups, PRICING, threshold_type="relative", threshold_value=0.9,
                                min_samples=3, edges=EDGES)
    assert bucket_key(one_call_group("x", CLEAN_INPUT), EDGES) not in routes


def test_fit_bucket_routes_computes_edges_itself_when_not_given():
    groups = {**repeat_groups("claude-opus-5", CLEAN_INPUT, 3, "a"),
              **repeat_groups("claude-sonnet-5", CLEAN_INPUT, 3, "b")}
    routes = fit_bucket_routes(groups, PRICING, threshold_type="relative", threshold_value=0.9)
    assert len(routes) == 1  # both models' identical-shaped calls land in the one bucket compute_edges derives


# ---- make_route_trajectory ----

def test_make_route_trajectory_uses_the_learned_bucket_model():
    bucket_routes = {bucket_key(one_call_group("x", CLEAN_INPUT), EDGES): "claude-sonnet-5"}
    route_trajectory = make_route_trajectory(bucket_routes, EDGES)
    calls = one_call_group("claude-opus-5", CLEAN_INPUT)
    assert route_trajectory(calls) == ["claude-sonnet-5"]


def test_make_route_trajectory_falls_back_to_logged_model_for_unknown_bucket():
    route_trajectory = make_route_trajectory({}, EDGES)
    calls = one_call_group("claude-opus-5", CLEAN_INPUT)
    assert route_trajectory(calls) == ["claude-opus-5"]
