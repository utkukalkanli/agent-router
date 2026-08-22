#!/usr/bin/env python3
"""Routemaxxing: an interpretable, family-preserving model router and honest OPE.

The real export behaves like one complete agent history per JSONL row.  We unfold each
history into the model calls it contains, retain the official "final output missing"
interpretation for the primary estimate, and keep the observed terminal assistant text
as an explicitly labelled sensitivity signal.

The certified policy chooses one model for a whole trajectory.  The adaptive policy may
escalate once after an observed error/retry/context-growth signal, never downgrade, and
is reported as conditional on logged states rather than as an identified causal effect.

No API key, GPU, sklearn, or measured token usage is required.  All token counts are
estimated as serialized JSON characters / 4.

Usage:
    python scripts/routemaxxing.py export/

Writes results/routemaxxing/ plus results/routemaxxing.jsonl for the explorer join.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cost_model import load_pricing, price_of  # noqa: E402
from decompose import decompose  # noqa: E402
from load_trajectories import est_tokens  # noqa: E402
from trajectory_ids import extract, first_user_text, path_is_collapsed, sha  # noqa: E402


# Frozen before the team-name change.  Branding must not reshuffle train/validation
# membership after results have been inspected.
SEED = bytes.fromhex("726f757465726d61787865722d76312d3230323630383232").decode()
QUALITY_MARGIN = 0.02
MIN_MODEL_OBS = 20
MIN_EFFECTIVE_N = 15
MIN_PROPENSITY = 0.05
BOOTSTRAPS = 2_000

FAMILY_CANDIDATES = {
    "claude": ("claude-sonnet-5", "claude-opus-5", "claude-fable-5"),
    "gpt": ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"),
}

ERROR_RE = re.compile(
    r"(?:\berror\b|\bfailed\b|\bfailure\b|\bexception\b|traceback|"
    r"permission denied|unauthorized|invalid request|timed?\s*out|connection refused|"
    r"not found)", re.I,
)
FAILURE_RE = re.compile(
    r"(?:\bi(?:'m| am)? sorry\b.{0,80}\b(?:cannot|can't|unable|failed)\b|"
    r"\b(?:cannot|can't|unable to|could not|couldn't|failed to)\b|"
    r"\b(?:task|request|operation) (?:failed|was not completed)\b)", re.I | re.S,
)
CODE_RE = re.compile(r"\b(code|repo|repository|python|typescript|bug|test|deploy|file|patch)\b", re.I)
MESSAGE_RE = re.compile(r"\b(slack|teams|message|channel|coworker|reply|notify)\b", re.I)
READ_RE = re.compile(r"\b(read|search|find|summari[sz]e|report|audit|inspect|review)\b", re.I)
SIDE_EFFECT_RE = re.compile(r"\b(send|delete|upload|deploy|publish|write|edit|patch|remove)\b", re.I)
HIGH_STAKES_RE = re.compile(r"\b(legal|medical|security|permission|credential|invoice|payment)\b", re.I)
TRIGGER_RE = re.compile(r"^\s*-\s*Triggered by:\s*`([^`]+)`", re.M)


def _text(item: Dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict)
            and part.get("type") in ("input_text", "output_text", "text")
        )
    return ""


def _tool_name(item: Dict[str, Any]) -> Optional[str]:
    return item.get("name") or (item.get("function") or {}).get("name")


def _tool_output_text(item: Dict[str, Any]) -> str:
    value = item.get("output", item.get("content", ""))
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _family(model: str) -> str:
    return "claude" if model.startswith("claude") else "gpt"


def infer_family(tools: Sequence[Dict[str, Any]], default_model: str = "") -> str:
    """Infer the compatible model family from the pre-call tool dialect."""
    names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    if names & {"bash", "file_edit", "file_write", "file_read"}:
        return "claude"
    if names & {"shell_command", "apply_patch"}:
        return "gpt"
    return _family(default_model) if default_model else "unknown"


@dataclass
class Call:
    index: int
    input_items: List[Dict[str, Any]]
    observed_output: List[Dict[str, Any]]
    environment_delta: List[Dict[str, Any]]
    input_tokens: int
    tools_tokens: int
    output_tokens: int
    cacheable_tokens: int
    is_final: bool


@dataclass
class Outcome:
    terminal_language: bool
    unresolved_tool_error: bool
    retry_loop: bool

    @property
    def observed_failure(self) -> int:
        return int(self.terminal_language or self.unresolved_tool_error or self.retry_loop)

    @property
    def official_failure(self) -> int:
        return int(self.unresolved_tool_error or self.retry_loop)


@dataclass
class Trajectory:
    row: int
    trajectory_id: str
    logged_model: str
    family: str
    items: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    calls: List[Call]
    opening: str
    group_key: str
    split: str
    features: List[float]
    outcome: Outcome
    feature_values: Dict[str, float] = field(default_factory=dict)


FEATURE_NAMES = (
    "bias", "log_initial_tokens", "tool_count", "has_image", "trigger_cron",
    "trigger_user", "intent_code", "intent_message", "intent_read",
    "intent_side_effect", "intent_high_stakes",
)


def _has_image(items: Sequence[Dict[str, Any]]) -> bool:
    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("input_image", "image_url")
            for part in content
        ):
            return True
    return False


def _base_features(opening: str, first_input: Sequence[Dict[str, Any]], tools: Sequence[Dict[str, Any]]) -> Tuple[List[float], Dict[str, float]]:
    trigger_match = TRIGGER_RE.search(opening)
    trigger = trigger_match.group(1).lower() if trigger_match else ""
    values = {
        "bias": 1.0,
        "log_initial_tokens": min(math.log1p(est_tokens(list(first_input)) + est_tokens(list(tools))) / 12.0, 1.0),
        "tool_count": min(len(tools) / 20.0, 1.0),
        "has_image": float(_has_image(first_input)),
        "trigger_cron": float(trigger == "cron"),
        "trigger_user": float(bool(trigger) and trigger not in ("cron", "trigger")),
        "intent_code": float(bool(CODE_RE.search(opening))),
        "intent_message": float(bool(MESSAGE_RE.search(opening))),
        "intent_read": float(bool(READ_RE.search(opening))),
        "intent_side_effect": float(bool(SIDE_EFFECT_RE.search(opening))),
        "intent_high_stakes": float(bool(HIGH_STAKES_RE.search(opening))),
    }
    return [values[name] for name in FEATURE_NAMES], values


def _split(group_key: str) -> str:
    value = int(hashlib.sha256((SEED + ":" + group_key).encode()).hexdigest()[:8], 16)
    return "train" if value % 100 < 70 else "validation"


def outcome_from_items(items: Sequence[Dict[str, Any]]) -> Outcome:
    assistants = [
        item for item in items if isinstance(item, dict)
        and item.get("role") == "assistant"
    ]
    final_text = _text(assistants[-1]) if assistants else ""
    terminal_language = bool(FAILURE_RE.search(final_text))

    outputs = [
        (index, item) for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") in (
            "function_call_output", "custom_tool_call_output",
        )
    ]
    unresolved_tool_error = bool(outputs and ERROR_RE.search(_tool_output_text(outputs[-1][1])))

    signatures: List[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in (
            "function_call", "custom_tool_call",
        ):
            continue
        args = item.get("arguments", item.get("input", ""))
        if not isinstance(args, str):
            args = json.dumps(args, sort_keys=True)
        signatures.append("%s:%s" % (_tool_name(item), re.sub(r"\s+", " ", args).strip()))
    run = maximum = 0
    previous = None
    for signature in signatures:
        run = run + 1 if signature == previous else 1
        maximum = max(maximum, run)
        previous = signature
    return Outcome(terminal_language, unresolved_tool_error, maximum >= 3)


def trajectory_from_row(row: Dict[str, Any], row_index: int) -> Trajectory:
    items = row.get("input") or []
    tools = row.get("tools") or []
    spans = decompose(items)
    if not spans:
        raise ValueError("row %d has no model-authored call" % row_index)
    calls: List[Call] = []
    tools_tokens = est_tokens(tools)
    for index, (start, stop) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(items)
        input_items = items[:start]
        previous_start = spans[index - 1][0] if index else None
        # Match cost_model.shared_prefix_tokens and the explorer exactly: cacheable
        # history is the sum of equal item estimates, not a newly serialized list
        # (whose commas/brackets would add a small, artificial discrepancy).
        cacheable = sum(est_tokens(item) for item in items[:previous_start]) + tools_tokens if previous_start is not None else 0
        calls.append(Call(
            index=index,
            input_items=input_items,
            observed_output=items[start:stop],
            environment_delta=items[stop:next_start],
            input_tokens=est_tokens(input_items),
            tools_tokens=tools_tokens,
            output_tokens=est_tokens(items[start:stop]),
            cacheable_tokens=cacheable,
            is_final=index == len(spans) - 1,
        ))
    trajectory_id = sha(items, 12)
    opening = first_user_text(row)
    meta = extract(row)
    clean_thread = None if path_is_collapsed(meta["thread_path"]) else meta["thread_path"]
    grouping = "%s:%s" % (meta["tenant"], clean_thread) if clean_thread else trajectory_id
    features, values = _base_features(opening, calls[0].input_items, tools)
    logged = row["model"]
    inferred = infer_family(tools, logged)
    family = _family(logged)
    if inferred not in (family, "unknown"):
        raise ValueError("tool dialect/model family mismatch in row %d" % row_index)
    return Trajectory(
        row=row_index, trajectory_id=trajectory_id, logged_model=logged, family=family,
        items=items, tools=tools, calls=calls, opening=opening, group_key=grouping,
        split=_split(grouping), features=features, feature_values=values,
        outcome=outcome_from_items(items),
    )


def load_export(export: str) -> List[Trajectory]:
    paths = sorted(Path(export).glob("*.jsonl")) if Path(export).is_dir() else [Path(export)]
    trajectories: List[Trajectory] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trajectories.append(trajectory_from_row(json.loads(line), len(trajectories)))
    return trajectories


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -40.0))
    return z / (1.0 + z)


class BinaryLogit:
    """Small deterministic L2 logistic regression for the offline laptop path."""

    def __init__(self, l2: float = 2.0, iterations: int = 1_200):
        self.l2, self.iterations, self.weights = l2, iterations, []

    def fit(self, xs: Sequence[Sequence[float]], ys: Sequence[int]) -> "BinaryLogit":
        if not xs:
            raise ValueError("cannot fit empty outcome model")
        self.weights = [0.0] * len(xs[0])
        n = len(xs)
        for step in range(self.iterations):
            gradient = [0.0] * len(self.weights)
            for x, y in zip(xs, ys):
                error = _sigmoid(sum(w * v for w, v in zip(self.weights, x))) - y
                for j, value in enumerate(x):
                    gradient[j] += error * value
            rate = 0.35 / math.sqrt(1.0 + step / 80.0)
            for j in range(len(self.weights)):
                penalty = 0.0 if j == 0 else self.l2 * self.weights[j]
                self.weights[j] -= rate * (gradient[j] + penalty) / n
        return self

    def predict(self, x: Sequence[float]) -> float:
        return min(max(_sigmoid(sum(w * v for w, v in zip(self.weights, x))), 0.001), 0.999)


class Softmax:
    """Deterministic multinomial propensity model."""

    def __init__(self, labels: Sequence[str], l2: float = 2.0, iterations: int = 1_000):
        self.labels = list(labels)
        self.l2, self.iterations, self.weights = l2, iterations, []

    def fit(self, xs: Sequence[Sequence[float]], ys: Sequence[str]) -> "Softmax":
        self.weights = [[0.0] * len(xs[0]) for _ in self.labels]
        n = len(xs)
        for step in range(self.iterations):
            gradients = [[0.0] * len(xs[0]) for _ in self.labels]
            for x, y in zip(xs, ys):
                probabilities = self.predict(x)
                for k, label in enumerate(self.labels):
                    error = probabilities[label] - float(y == label)
                    for j, value in enumerate(x):
                        gradients[k][j] += error * value
            rate = 0.3 / math.sqrt(1.0 + step / 80.0)
            for k in range(len(self.labels)):
                for j in range(len(xs[0])):
                    penalty = 0.0 if j == 0 else self.l2 * self.weights[k][j]
                    self.weights[k][j] -= rate * (gradients[k][j] + penalty) / n
        return self

    def predict(self, x: Sequence[float]) -> Dict[str, float]:
        scores = [sum(w * v for w, v in zip(row, x)) for row in self.weights]
        top = max(scores)
        exp = [math.exp(value - top) for value in scores]
        total = sum(exp)
        return {label: value / total for label, value in zip(self.labels, exp)}


@dataclass
class FamilyModel:
    candidates: List[str]
    outcome_model: BinaryLogit
    propensity_model: Softmax
    counts: Dict[str, int]
    context_p90: int

    def augmented(self, base: Sequence[float], model: str) -> List[float]:
        values = list(base)
        for candidate in self.candidates[1:]:
            active = float(model == candidate)
            values.append(active)
            values.extend(active * value for value in base[1:])
        return values

    def risk(self, trajectory: Trajectory, model: str) -> float:
        return self.outcome_model.predict(self.augmented(trajectory.features, model))

    def propensity(self, trajectory: Trajectory) -> Dict[str, float]:
        return self.propensity_model.predict(trajectory.features)


def fit_models(trajectories: Sequence[Trajectory], official: bool = False) -> Dict[str, FamilyModel]:
    fitted: Dict[str, FamilyModel] = {}
    for family, configured in FAMILY_CANDIDATES.items():
        rows = [t for t in trajectories if t.split == "train" and t.family == family]
        counts = Counter(t.logged_model for t in rows)
        candidates = [model for model in configured if counts[model] >= MIN_MODEL_OBS]
        if len(candidates) < 2:
            raise ValueError("not enough supported %s models: %r" % (family, counts))
        eligible = [t for t in rows if t.logged_model in candidates]
        template = FamilyModel(candidates, BinaryLogit(), Softmax(candidates), dict(counts), 0)
        xs = [template.augmented(t.features, t.logged_model) for t in eligible]
        ys = [t.outcome.official_failure if official else t.outcome.observed_failure for t in eligible]
        outcome_model = BinaryLogit().fit(xs, ys)
        propensity = Softmax(candidates).fit(
            [t.features for t in eligible], [t.logged_model for t in eligible],
        )
        contexts = sorted(call.input_tokens + call.tools_tokens for t in eligible for call in t.calls)
        p90 = contexts[min(int(0.9 * len(contexts)), len(contexts) - 1)]
        fitted[family] = FamilyModel(candidates, outcome_model, propensity, dict(counts), p90)
    return fitted


def choose_initial(trajectory: Trajectory, model: FamilyModel, pricing: Dict[str, Sequence[float]], margin: float = QUALITY_MARGIN) -> Tuple[str, List[str], Dict[str, float]]:
    risks = {candidate: model.risk(trajectory, candidate) for candidate in model.candidates}
    propensities = model.propensity(trajectory)
    best = min(risks.values())
    # ``default_model`` is part of the public decision context.  The certified
    # policy is a downgrade-or-keep policy: it never spends more than the configured
    # default merely because an observational model predicts a small quality gain.
    default_price = price_of(trajectory.logged_model, pricing)[0]
    ordered = sorted(
        (candidate for candidate in model.candidates
         if price_of(candidate, pricing)[0] <= default_price),
        key=lambda candidate: price_of(candidate, pricing)[0],
    )
    for candidate in ordered:
        if propensities.get(candidate, 0.0) >= MIN_PROPENSITY and risks[candidate] <= best + margin:
            reasons = ["cheapest-supported", "within-2pp-risk"]
            reasons.extend(name for name, value in trajectory.feature_values.items() if value and name.startswith("intent_"))
            return candidate, reasons, risks
    # Unsupported/legacy defaults are preserved exactly.  They remain valid models
    # and comparators; lack of support is not permission to synthesize an upgrade.
    return trajectory.logged_model, ["out-of-support-fallback"], risks


def _dynamic_flags(trajectory: Trajectory, call_index: int, context_p90: int) -> List[str]:
    if call_index <= 0:
        return []
    current = trajectory.calls[call_index]
    prior_items = current.input_items
    outputs = [
        item for item in prior_items if isinstance(item, dict)
        and item.get("type") in ("function_call_output", "custom_tool_call_output")
    ]
    flags = []
    if outputs and ERROR_RE.search(_tool_output_text(outputs[-1])):
        flags.append("prior-tool-error")
    signatures = []
    for item in prior_items:
        if isinstance(item, dict) and item.get("type") in ("function_call", "custom_tool_call"):
            signatures.append(_tool_name(item))
    if len(signatures) >= 2 and signatures[-1] == signatures[-2]:
        flags.append("repeated-tool")
    if current.input_tokens + current.tools_tokens > context_p90:
        flags.append("context-p90")
    if any(item.get("role") == "user" for item in trajectory.calls[call_index - 1].environment_delta if isinstance(item, dict)):
        flags.append("user-follow-up")
    return flags


def make_routes(trajectory: Trajectory, model: FamilyModel, pricing: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    initial, reasons, risks = choose_initial(trajectory, model, pricing)
    certified = [initial] * len(trajectory.calls)
    adaptive = [initial]
    adaptive_reasons = {0: list(reasons)}
    switched = False
    current = initial
    for index in range(1, len(trajectory.calls)):
        flags = _dynamic_flags(trajectory, index, model.context_p90)
        hard_failure = bool({"prior-tool-error", "repeated-tool"} & set(flags))
        risky_growth = "context-p90" in flags and bool(
            trajectory.feature_values["intent_high_stakes"]
            or trajectory.feature_values["intent_side_effect"]
        )
        if (hard_failure or risky_growth) and not switched:
            current_price = price_of(current, pricing)[0]
            default_price = price_of(trajectory.logged_model, pricing)[0]
            stronger = [
                candidate for candidate in model.candidates
                if current_price < price_of(candidate, pricing)[0] <= default_price
            ]
            if stronger:
                best = min(model.risk(trajectory, candidate) for candidate in stronger)
                supported = [
                    candidate for candidate in stronger
                    if model.propensity(trajectory).get(candidate, 0.0) >= MIN_PROPENSITY
                    and model.risk(trajectory, candidate) <= best + QUALITY_MARGIN
                ]
                if supported:
                    current = min(supported, key=lambda candidate: price_of(candidate, pricing)[0])
                    switched = True
                    adaptive_reasons[index] = ["one-way-escalation"] + flags
        adaptive.append(current)
    return {
        "certified": certified, "adaptive": adaptive, "initial_reasons": reasons,
        "adaptive_reasons": adaptive_reasons, "risks": risks,
    }


def route_cost(trajectory: Trajectory, route: Sequence[str], pricing: Dict[str, Sequence[float]], include_observed_final: bool = False, output_ratios: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    total = switch_penalty = 0.0
    for index, (call, selected) in enumerate(zip(trajectory.calls, route)):
        same = index > 0 and route[index - 1] == selected
        cacheable = min(call.cacheable_tokens, call.input_tokens + call.tools_tokens) if same else 0
        pu, pc, po = price_of(selected, pricing)
        billed = call.input_tokens + call.tools_tokens
        output_tokens = call.output_tokens if (include_observed_final or not call.is_final) else 0
        if output_ratios and selected != trajectory.logged_model:
            denominator = max(output_ratios.get(trajectory.logged_model, 1.0), 0.1)
            output_tokens = int(output_tokens * output_ratios.get(selected, 1.0) / denominator)
        total += ((billed - cacheable) * pu + cacheable * pc + output_tokens * po) / 1e6
        if index > 0 and route[index - 1] != selected:
            switch_penalty += call.cacheable_tokens * (pu - pc) / 1e6
    return total, switch_penalty


def output_length_ratios(trajectories: Sequence[Trajectory]) -> Dict[str, float]:
    by_model: Dict[str, List[float]] = {}
    for trajectory in trajectories:
        if trajectory.split != "train":
            continue
        values = [call.output_tokens for call in trajectory.calls if not call.is_final]
        if values:
            by_model.setdefault(trajectory.logged_model, []).append(sum(values) / len(values))
    means = {model: statistics.fmean(values) for model, values in by_model.items()}
    ratios: Dict[str, float] = {}
    for family in FAMILY_CANDIDATES:
        family_values = [value for model, value in means.items() if _family(model) == family]
        median = statistics.median(family_values) if family_values else 1.0
        for model, value in means.items():
            if _family(model) == family:
                ratios[model] = min(max(value / median, 0.5), 2.0)
    return ratios


def _effective_n(weights: Sequence[float]) -> float:
    return (sum(weights) ** 2 / sum(value * value for value in weights)) if weights else 0.0


def estimate_policy(trajectories: Sequence[Trajectory], fitted: Dict[str, FamilyModel], actions: Dict[str, str], official: bool, seed_suffix: str = "") -> Dict[str, float]:
    """Estimate downgrade-from-default risk in common support.

    The target policy accepts ``default_model`` as an input and either keeps it or
    downgrades it.  That is an ATT-style estimand, not a treatment policy independent
    of the logged action, so a vanilla AIPW equation is inappropriate.  We use the
    held-out outcome regression for the risk contrast, propensity-screen the proposed
    alternate, and cluster-bootstrap trajectories.  Unchanged decisions contribute
    exactly zero by construction.
    """
    contributions: List[float] = []
    weights: List[float] = []
    covered = 0
    for trajectory in trajectories:
        model = fitted[trajectory.family]
        if trajectory.logged_model not in model.candidates:
            continue
        action = actions.get(trajectory.trajectory_id, trajectory.logged_model)
        if action not in model.candidates:
            action = trajectory.logged_model
        propensities = model.propensity(trajectory)
        action_propensity = propensities.get(action, 0.0)
        if action_propensity < MIN_PROPENSITY:
            action = trajectory.logged_model
        else:
            covered += int(action != trajectory.logged_model)
        mu_policy = model.risk(trajectory, action)
        mu_logged = model.risk(trajectory, trajectory.logged_model)
        if action == trajectory.logged_model:
            contributions.append(0.0)
        else:
            contributions.append(mu_policy - mu_logged)
            weights.append(1.0 / max(propensities[action], MIN_PROPENSITY))
    estimate = statistics.fmean(contributions) if contributions else 0.0
    rng = random.Random(SEED + seed_suffix)
    boots = []
    for _ in range(BOOTSTRAPS):
        sample = [contributions[rng.randrange(len(contributions))] for _ in contributions]
        boots.append(statistics.fmean(sample))
    boots.sort()
    upper = boots[min(int(0.95 * len(boots)), len(boots) - 1)] if boots else estimate
    return {
        "failure_delta": estimate,
        "failure_upper95": upper,
        "evaluation_n": len(contributions),
        "routed_coverage": covered / len(contributions) if contributions else 0.0,
        "effective_n": _effective_n(weights) if weights else len(contributions),
    }


def _proposal_priority(trajectory: Trajectory, route_data: Dict[str, Any], pricing: Dict[str, Sequence[float]]) -> Tuple[float, float]:
    initial = route_data["certified"][0]
    risks = route_data["risks"]
    risk_delta = risks[initial] - risks.get(trajectory.logged_model, risks[initial])
    logged, _ = route_cost(trajectory, [trajectory.logged_model] * len(trajectory.calls), pricing)
    routed, _ = route_cost(trajectory, route_data["certified"], pricing)
    return risk_delta, -(logged - routed)


def _adopt_actions(trajectories: Sequence[Trajectory], proposals: Dict[str, Dict[str, Any]], pricing: Dict[str, Sequence[float]], fraction: float) -> Dict[str, str]:
    changed = [t for t in trajectories if proposals[t.trajectory_id]["certified"][0] != t.logged_model]
    changed.sort(key=lambda t: _proposal_priority(t, proposals[t.trajectory_id], pricing))
    take = int(round(fraction * len(changed)))
    adopted = {t.trajectory_id for t in changed[:take]}
    return {
        t.trajectory_id: proposals[t.trajectory_id]["certified"][0] if t.trajectory_id in adopted else t.logged_model
        for t in trajectories
    }


def _frontier(trajectories: Sequence[Trajectory], fitted: Dict[str, FamilyModel], official_fitted: Dict[str, FamilyModel], proposals: Dict[str, Dict[str, Any]], pricing: Dict[str, Sequence[float]], output_ratios: Dict[str, float], selected_fraction: Optional[float] = None) -> Tuple[List[Dict[str, Any]], float]:
    rows = []
    for step in range(11):
        fraction = step / 10.0
        actions = _adopt_actions(trajectories, proposals, pricing, fraction)
        quality = estimate_policy(trajectories, fitted, actions, official=False, seed_suffix="-%s" % fraction)
        official = estimate_policy(trajectories, official_fitted, actions, official=True, seed_suffix="-official-%s" % fraction)
        logged_cost = routed_cost = adjusted_cost = 0.0
        for trajectory in trajectories:
            logged_route = [trajectory.logged_model] * len(trajectory.calls)
            route = [actions[trajectory.trajectory_id]] * len(trajectory.calls)
            logged_cost += route_cost(trajectory, logged_route, pricing)[0]
            routed_cost += route_cost(trajectory, route, pricing)[0]
            adjusted_cost += route_cost(trajectory, route, pricing, output_ratios=output_ratios)[0]
        rows.append({
            "adopt_frac": fraction,
            "cost_usd_est": round(routed_cost, 4),
            "behavior_adjusted_cost_usd_est": round(adjusted_cost, 4),
            "savings_pct": round((1.0 - routed_cost / logged_cost) * 100.0, 2) if logged_cost else 0.0,
            "proxy_quality": round(1.0 - quality["failure_delta"], 4),
            "failure_delta": round(quality["failure_delta"], 5),
            "failure_upper95": round(quality["failure_upper95"], 5),
            "official_failure_delta": round(official["failure_delta"], 5),
            "official_failure_upper95": round(official["failure_upper95"], 5),
            "coverage": round(quality["routed_coverage"], 4),
            "effective_n": round(quality["effective_n"], 1),
            "selected": False,
        })
    if selected_fraction is None:
        eligible = [row for row in rows if row["failure_upper95"] <= QUALITY_MARGIN]
        selected = min(eligible, key=lambda row: row["cost_usd_est"]) if eligible else rows[0]
        selected_fraction = selected["adopt_frac"]
    selected = min(rows, key=lambda row: abs(row["adopt_frac"] - selected_fraction))
    selected["selected"] = True
    return rows, selected["adopt_frac"]


def corrected_baseline(trajectory: Trajectory) -> List[str]:
    total = sum(call.input_tokens + call.tools_tokens for call in trajectory.calls)
    if total >= 15_000:
        return [trajectory.logged_model] * len(trajectory.calls)
    cheap = "claude-sonnet-5" if trajectory.family == "claude" else "gpt-5.6-luna"
    return [cheap] * len(trajectory.calls)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _frontier_svg(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    width, height = 960, 540
    left, right, top, bottom = 100, 45, 55, 85
    costs = [row["cost_usd_est"] for row in rows]
    quality = [1.0 - row["failure_delta"] for row in rows]
    min_x, max_x = min(costs), max(costs)
    min_y, max_y = min(quality), max(quality)
    xpad = max((max_x - min_x) * 0.08, 0.01)
    ypad = max((max_y - min_y) * 0.3, 0.005)
    min_x, max_x = min_x - xpad, max_x + xpad
    min_y, max_y = min_y - ypad, max_y + ypad
    def x(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * (width - left - right)
    def y(value: float) -> float:
        return height - bottom - (value - min_y) / (max_y - min_y) * (height - top - bottom)
    points = " ".join("%.1f,%.1f" % (x(cx), y(cy)) for cx, cy in zip(costs, quality))
    selected = next(row for row in rows if row["selected"])
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%%" height="100%%" fill="#fbfaff"/>',
        '<text x="%d" y="30" fill="#150079" font-family="system-ui" font-size="22" font-weight="700">Public validation frontier · estimated tokens, cache-aware</text>' % left,
    ]
    for index in range(6):
        yy_value = min_y + (max_y - min_y) * index / 5
        yy = y(yy_value)
        parts.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#ded8ff"/>' % (left, width - right, yy, yy))
        parts.append('<text x="%d" y="%.1f" text-anchor="end" fill="#150079" font-family="system-ui" font-size="14">%.3f</text>' % (left - 10, yy + 5, yy_value))
    for index in range(6):
        xx_value = min_x + (max_x - min_x) * index / 5
        xx = x(xx_value)
        parts.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="#eeeaff"/>' % (xx, xx, top, height - bottom))
        parts.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#150079" font-family="system-ui" font-size="14">$%.1f</text>' % (xx, height - bottom + 25, xx_value))
    parts.append('<polyline points="%s" fill="none" stroke="#6748FD" stroke-width="5"/>' % points)
    for row, cx, cy in zip(rows, costs, quality):
        fill = "#FFBD9E" if row["selected"] else "#6748FD"
        radius = 10 if row["selected"] else 6
        parts.append('<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" stroke="#150079" stroke-width="2"/>' % (x(cx), y(cy), radius, fill))
    parts.extend([
        '<text x="%d" y="%d" text-anchor="middle" fill="#150079" font-family="system-ui" font-size="17">Cost (USD estimate; final output excluded)</text>' % ((left + width - right) // 2, height - 28),
        '<text transform="translate(26 %d) rotate(-90)" text-anchor="middle" fill="#150079" font-family="system-ui" font-size="17">1 − estimated failure delta</text>' % ((top + height - bottom) // 2),
        '<text x="%.1f" y="%.1f" fill="#150079" font-family="system-ui" font-size="15" font-weight="700">selected: %.0f%% proposal adoption</text>' % (x(selected["cost_usd_est"]) + 14, y(1.0 - selected["failure_delta"]) + 32, selected["adopt_frac"] * 100),
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def _presentation(path: Path, summary: Dict[str, Any], selected: Dict[str, Any]) -> None:
    claim = (
        "Public validation: %+.1f%% estimated cost at a %.1f pp proxy-failure upper bound"
        % (-selected["savings_pct"], selected["failure_upper95"] * 100)
        if selected["adopt_frac"] > 0 else
        "We found 10,845 model turns hiding inside 1,000 exported rows"
    )
    document = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>routemaxxing — Viktor Challenge</title>
<style>
:root{--violet:#6748FD;--navy:#150079;--peach:#FFBD9E;--ink:#0d0d1a;--paper:#fbfaff}*{margin:0;padding:0;box-sizing:border-box}body{background:var(--ink);font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif}.slide{display:none;width:100vw;height:100vh;padding:7vh 8vw;background:var(--paper);color:var(--navy);flex-direction:column;justify-content:center}.slide.active{display:flex}.slide.dark{background:var(--navy);color:var(--paper)}h1{font-size:5vw;line-height:1.05}h2{font-size:3.2vw;margin-bottom:4vh}p,li{font-size:1.55vw;line-height:1.45;max-width:76vw}.kicker{color:var(--violet);text-transform:uppercase;letter-spacing:.18em;font-size:1vw;font-weight:700;margin-bottom:2.5vh}.dark .kicker{color:var(--peach)}.big{font-size:2.6vw;font-weight:700;color:var(--violet);margin:2vh 0}.dark .big{color:var(--peach)}.cols{display:flex;gap:3vw;margin-top:2vh}.card{flex:1;border:2px solid var(--violet);border-radius:14px;padding:2.5vh 1.8vw}.card h3{color:var(--violet);font-size:1.35vw;margin-bottom:1vh}.chart{height:59vh;width:auto;max-width:84vw}.foot{position:fixed;bottom:2.5vh;right:8vw;font-size:.9vw;color:var(--violet);opacity:.7}code{background:rgba(103,72,253,.1);padding:.1em .35em;border-radius:5px}
</style></head><body>
<section class="slide dark active"><div class="kicker">Viktor Challenge · TUM.ai · Build the Router</div><h1>routemaxxing</h1><p class="big">%s</p><p>Atahan Kılıç · Musa Kocabasoğlu · Utku Kalkanlı</p></section>
<section class="slide"><div class="kicker">Objective</div><h2>Cheapest route that clears an honest quality bar</h2><p>Minimize cache-aware estimated dollars while the one-sided 95%% upper bound on proxy-failure degradation stays within <b>2 percentage points</b>.</p></section>
<section class="slide"><div class="kicker">The signal</div><h2>Fix the unit before routing it</h2><div class="cols"><div class="card"><h3>Structure in the traces</h3><p><b>1,000 unique histories → 10,845 model turns.</b> Tool dialect locks the family; opening intent sets the tier; prior errors, retries and context growth can trigger one escalation.</p></div><div class="card"><h3>A real decision boundary</h3><p><code>bash + file_edit</code> stays in Claude. <code>shell_command + apply_patch</code> stays in GPT. We never estimate a cross-family effect where the log has zero overlap.</p></div></div></section>
<section class="slide"><div class="kicker">The one chart</div><h2>Cost–proxy-quality frontier <span style="font-size:.5em">(public validation; held-out pending)</span></h2><img class="chart" src="results/routemaxxing/frontier.svg" alt="frontier"></section>
<section class="slide"><div class="kicker">Honesty</div><h2>What the log can—and cannot—identify</h2><ul style="list-style:none"><li><b style="color:var(--violet)">Method —</b> Family-restricted, propensity-screened outcome regression for downgrade-from-default effects; 2,000 trajectory bootstraps.</li><li style="margin-top:2vh"><b style="color:var(--violet)">Cache trap —</b> Same-model prefixes use cache-read prices. The single allowed escalation rebills the accumulated prefix uncached.</li><li style="margin-top:2vh"><b style="color:var(--violet)">Weakest point —</b> Quality is an automatic proxy, and a different early model could create a different later context. Adaptive results are conditional, not causal.</li></ul></section>
<section class="slide dark"><div class="kicker">Close</div><h1 style="font-size:3.5vw">A router is only as honest as its unit of analysis.</h1><p class="big">%s trajectories · %s turns · %.1f%% validation routed coverage</p><p>Next: run the locked policy once on the organizer-held-out split; optionally judge-replay 120 calls near the escalation boundary.</p></section>
<div class="foot">routemaxxing · Viktor challenge · TUM.ai</div><script>const s=[...document.querySelectorAll('.slide')];let i=0;const show=n=>{s[i].classList.remove('active');i=(n+s.length)%%s.length;s[i].classList.add('active')};addEventListener('keydown',e=>{if(['ArrowRight',' ','PageDown'].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key==='f')document.documentElement.requestFullscreen()});addEventListener('click',()=>show(i+1));</script></body></html>""" % (
        html.escape(claim), summary["trajectories"], f'{summary["calls"]:,}',
        selected["coverage"] * 100,
    )
    path.write_text(document, encoding="utf-8")


def run(export: str = "export", results_root: Path = ROOT / "results") -> Dict[str, Any]:
    pricing = load_pricing()
    trajectories = load_export(export)
    fitted = fit_models(trajectories, official=False)
    official_fitted = fit_models(trajectories, official=True)
    ratios = output_length_ratios(trajectories)
    proposals = {t.trajectory_id: make_routes(t, fitted[t.family], pricing) for t in trajectories}
    train = [t for t in trajectories if t.split == "train"]
    validation = [t for t in trajectories if t.split == "validation"]
    train_frontier, selected_fraction = _frontier(
        train, fitted, official_fitted, proposals, pricing, ratios,
    )
    frontier, _ = _frontier(
        validation, fitted, official_fitted, proposals, pricing, ratios, selected_fraction,
    )

    outdir = Path(results_root) / "routemaxxing"
    outdir.mkdir(parents=True, exist_ok=True)
    selected_actions = _adopt_actions(trajectories, proposals, pricing, selected_fraction)
    route_records = []
    adaptive_records = []
    ledger = []
    totals = Counter()
    for trajectory in trajectories:
        proposal = proposals[trajectory.trajectory_id]
        chosen = selected_actions[trajectory.trajectory_id]
        certified = [chosen] * len(trajectory.calls)
        adaptive = proposal["adaptive"] if chosen != trajectory.logged_model else certified
        logged_route = [trajectory.logged_model] * len(trajectory.calls)
        logged_cost, _ = route_cost(trajectory, logged_route, pricing)
        certified_cost, certified_penalty = route_cost(trajectory, certified, pricing)
        adaptive_cost, adaptive_penalty = route_cost(trajectory, adaptive, pricing)
        adjusted_cost, _ = route_cost(trajectory, certified, pricing, output_ratios=ratios)
        baseline = corrected_baseline(trajectory)
        baseline_cost, _ = route_cost(trajectory, baseline, pricing)
        totals.update({
            "logged_micros": round(logged_cost * 1e6),
            "certified_micros": round(certified_cost * 1e6),
            "adaptive_micros": round(adaptive_cost * 1e6),
            "adjusted_micros": round(adjusted_cost * 1e6),
            "baseline_micros": round(baseline_cost * 1e6),
        })
        common = {
            "trajectory_id": trajectory.trajectory_id,
            "trajectory": trajectory.trajectory_id,
            "n_calls": len(trajectory.calls),
            "logged_model": trajectory.logged_model,
            "split": trajectory.split,
            "proxy_failure": trajectory.outcome.observed_failure,
            "official_proxy_failure": trajectory.outcome.official_failure,
        }
        route_records.append({
            **common, "route": certified,
            "cost_logged_usd": round(logged_cost, 8),
            "cost_routed_usd": round(certified_cost, 8),
            "behavior_adjusted_cost_usd": round(adjusted_cost, 8),
            "switches": 0, "switch_penalty_usd": round(certified_penalty, 8),
            "reason_codes": proposal["initial_reasons"],
        })
        adaptive_records.append({
            **common, "route": adaptive,
            "cost_logged_usd": round(logged_cost, 8),
            "cost_routed_usd": round(adaptive_cost, 8),
            "switches": sum(adaptive[i] != adaptive[i - 1] for i in range(1, len(adaptive))),
            "switch_penalty_usd": round(adaptive_penalty, 8),
            "estimate_kind": "conditional-on-logged-states",
        })
        for call, selected in zip(trajectory.calls, certified):
            reasons = proposal["adaptive_reasons"].get(call.index, proposal["initial_reasons"] if call.index == 0 else [])
            ledger.append({
                "trajectory_id": trajectory.trajectory_id, "call_index": call.index,
                "split": trajectory.split, "logged_model": trajectory.logged_model,
                "chosen_model": selected, "adaptive_model": adaptive[call.index],
                "family": trajectory.family, "reason_codes": reasons,
                "risk_by_model": proposal["risks"],
                "propensity_by_model": fitted[trajectory.family].propensity(trajectory),
                "in_support": fitted[trajectory.family].propensity(trajectory).get(selected, 0) >= MIN_PROPENSITY,
                "switch_count": sum(adaptive[i] != adaptive[i - 1] for i in range(1, call.index + 1)),
                "input_tokens_est": call.input_tokens + call.tools_tokens,
                "output_tokens_est": None if call.is_final else call.output_tokens,
            })

    def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl(outdir / "routes.jsonl", route_records)
    write_jsonl(outdir / "adaptive_routes.jsonl", adaptive_records)
    write_jsonl(outdir / "ledger.jsonl", ledger)
    write_jsonl(Path(results_root) / "routemaxxing.jsonl", route_records)
    _write_csv(outdir / "train_frontier.csv", train_frontier)
    _write_csv(outdir / "frontier.csv", frontier)
    _frontier_svg(outdir / "frontier.svg", frontier)

    selected = next(row for row in frontier if row["selected"])
    logged = totals["logged_micros"] / 1e6
    summary = {
        "team": "routemaxxing",
        "members": ["Atahan Kılıç", "Musa Kocabasoğlu", "Utku Kalkanlı"],
        "data_interpretation": "one complete history per row; final output excluded in primary cost/outcome sensitivity",
        "trajectories": len(trajectories), "calls": sum(len(t.calls) for t in trajectories),
        "train_trajectories": len(train), "validation_trajectories": len(validation),
        "estimated_tokens": "serialized JSON characters / 4; not measured usage",
        "logged_cost_usd_est": round(logged, 4),
        "corrected_baseline_cost_usd_est": round(totals["baseline_micros"] / 1e6, 4),
        "certified_cost_usd_est": round(totals["certified_micros"] / 1e6, 4),
        "adaptive_cost_usd_est": round(totals["adaptive_micros"] / 1e6, 4),
        "behavior_adjusted_certified_cost_usd_est": round(totals["adjusted_micros"] / 1e6, 4),
        "selected_adopt_frac": selected_fraction,
        "validation_selected_point": selected,
        "quality_margin": QUALITY_MARGIN,
        "bootstrap_replicates": BOOTSTRAPS,
        "held_out_status": "not present; policy locked on public train and reported on public validation",
        "adaptive_estimate": "conditional on logged states; not an end-to-end causal claim",
        "known_weakness": "automatic proxy labels and counterfactual state drift after an alternative early model",
        "off_policy_method": "propensity-screened outcome regression for downgrade-from-default effects with trajectory bootstrap",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _presentation(ROOT / "presentation-routemaxxing.html", summary, selected)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export", help="export directory or JSONL path")
    args = parser.parse_args()
    summary = run(args.export)
    selected = summary["validation_selected_point"]
    print("rows=%d  reconstructed calls=%s" % (summary["trajectories"], f'{summary["calls"]:,}'))
    print("logged cost (cache-aware, input + observable output estimate): $%.4f" % summary["logged_cost_usd_est"])
    print("corrected heuristic baseline: $%.4f" % summary["corrected_baseline_cost_usd_est"])
    print("certified Routemaxxing:       $%.4f" % summary["certified_cost_usd_est"])
    print("adaptive Routemaxxing:        $%.4f (conditional estimate)" % summary["adaptive_cost_usd_est"])
    print("selected public-validation point: %.0f%% adoption, %.2f%% savings, failure upper95=%+.2f pp" % (
        selected["adopt_frac"] * 100, selected["savings_pct"], selected["failure_upper95"] * 100,
    ))
    print("NOTE: all tokens/costs are estimates; quality is an automatic proxy; held-out split pending.")


if __name__ == "__main__":
    main()
