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


V1_FEATURE_NAMES = (
    "bias", "log_initial_tokens", "tool_count", "has_image", "trigger_cron",
    "trigger_user", "intent_code", "intent_message", "intent_read",
    "intent_side_effect", "intent_high_stakes",
)

# All additions are available at call zero: no realized tool choice, future error,
# eventual depth, or final output enters the policy.
ENRICHED_FEATURE_NAMES = V1_FEATURE_NAMES + (
    "trigger_automation", "trigger_unknown",
    "size_xs", "size_s", "size_m", "size_l",
    "surface_slack", "surface_teams", "tool_memory", "tool_subagent",
    "tool_external_write",
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
    initial_tokens = est_tokens(list(first_input)) + est_tokens(list(tools))
    tool_names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    values = {
        "bias": 1.0,
        "log_initial_tokens": min(math.log1p(initial_tokens) / 12.0, 1.0),
        "tool_count": min(len(tools) / 20.0, 1.0),
        "has_image": float(_has_image(first_input)),
        "trigger_cron": float(trigger == "cron"),
        "trigger_user": float(bool(trigger) and trigger not in ("cron", "trigger")),
        "intent_code": float(bool(CODE_RE.search(opening))),
        "intent_message": float(bool(MESSAGE_RE.search(opening))),
        "intent_read": float(bool(READ_RE.search(opening))),
        "intent_side_effect": float(bool(SIDE_EFFECT_RE.search(opening))),
        "intent_high_stakes": float(bool(HIGH_STAKES_RE.search(opening))),
        "trigger_automation": float(trigger == "trigger"),
        "trigger_unknown": float(not trigger),
        "size_xs": float(initial_tokens < 10_000),
        "size_s": float(10_000 <= initial_tokens < 25_000),
        "size_m": float(25_000 <= initial_tokens < 50_000),
        "size_l": float(initial_tokens >= 50_000),
        "surface_slack": float(any("slack" in (name or "") for name in tool_names)),
        "surface_teams": float(any("msteams" in (name or "") for name in tool_names)),
        "tool_memory": float("memory_search" in tool_names),
        "tool_subagent": float("submit_subagent_result" in tool_names),
        "tool_external_write": float(any(
            token in (name or "") for name in tool_names
            for token in ("send_", "upload", "delete", "react")
        )),
    }
    return [values[name] for name in V1_FEATURE_NAMES], values


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
    feature_names: Tuple[str, ...] = V1_FEATURE_NAMES
    propensity_alpha: float = 0.0
    priors: Dict[str, float] = field(default_factory=dict)

    def vector(self, trajectory: Trajectory) -> List[float]:
        return [trajectory.feature_values[name] for name in self.feature_names]

    def augmented(self, base: Sequence[float], model: str) -> List[float]:
        values = list(base)
        for candidate in self.candidates[1:]:
            active = float(model == candidate)
            values.append(active)
            values.extend(active * value for value in base[1:])
        return values

    def risk(self, trajectory: Trajectory, model: str) -> float:
        return self.outcome_model.predict(self.augmented(self.vector(trajectory), model))

    def propensity(self, trajectory: Trajectory) -> Dict[str, float]:
        raw = self.propensity_model.predict(self.vector(trajectory))
        if not self.propensity_alpha:
            return raw
        # Dirichlet-style shrinkage toward the family model mix.  The local
        # pseudo-count is fixed before evaluation; alpha is an ablation parameter.
        local_strength = 20.0
        pooled = {
            candidate: (
                local_strength * raw[candidate]
                + self.propensity_alpha * self.priors[candidate]
            ) / (local_strength + self.propensity_alpha)
            for candidate in self.candidates
        }
        total = sum(pooled.values())
        return {candidate: value / total for candidate, value in pooled.items()}


def fit_models(
    trajectories: Sequence[Trajectory], official: bool = False,
    feature_names: Tuple[str, ...] = V1_FEATURE_NAMES,
    outcome_l2: float = 2.0, propensity_l2: float = 2.0,
    propensity_alpha: float = 0.0,
) -> Dict[str, FamilyModel]:
    fitted: Dict[str, FamilyModel] = {}
    for family, configured in FAMILY_CANDIDATES.items():
        rows = [t for t in trajectories if t.split == "train" and t.family == family]
        counts = Counter(t.logged_model for t in rows)
        candidates = [model for model in configured if counts[model] >= MIN_MODEL_OBS]
        if len(candidates) < 2:
            raise ValueError("not enough supported %s models: %r" % (family, counts))
        eligible = [t for t in rows if t.logged_model in candidates]
        priors = {candidate: counts[candidate] / len(eligible) for candidate in candidates}
        template = FamilyModel(
            candidates, BinaryLogit(), Softmax(candidates), dict(counts), 0,
            feature_names, propensity_alpha, priors,
        )
        xs = [template.augmented(template.vector(t), t.logged_model) for t in eligible]
        ys = [t.outcome.official_failure if official else t.outcome.observed_failure for t in eligible]
        outcome_model = BinaryLogit(l2=outcome_l2).fit(xs, ys)
        propensity = Softmax(candidates, l2=propensity_l2).fit(
            [template.vector(t) for t in eligible], [t.logged_model for t in eligible],
        )
        contexts = sorted(call.input_tokens + call.tools_tokens for t in eligible for call in t.calls)
        p90 = contexts[min(int(0.9 * len(contexts)), len(contexts) - 1)]
        fitted[family] = FamilyModel(
            candidates, outcome_model, propensity, dict(counts), p90,
            feature_names, propensity_alpha, priors,
        )
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


def choose_cost_effective_initial(
    trajectory: Trajectory, model: FamilyModel, pricing: Dict[str, Sequence[float]],
    output_ratios: Any, max_individual_risk: Optional[float] = None,
    savings_estimator: Optional["PredecisionSavings"] = None,
) -> Tuple[str, List[str], Dict[str, float], Dict[str, float]]:
    """Choose a supported, positive-savings downgrade relative to the default.

    Unlike the v1 gate, this compares risk with the configured default—not with an
    unrelated best model.  The global training-only frontier owns the 2pp constraint.
    """
    risks = {candidate: model.risk(trajectory, candidate) for candidate in model.candidates}
    metadata = {
        "fixed_savings": 0.0, "adjusted_savings": 0.0,
        "risk_delta": 0.0, "value_per_risk": 0.0,
    }
    if trajectory.logged_model not in model.candidates:
        return trajectory.logged_model, ["unsupported-default-fallback"], risks, metadata
    propensities = model.propensity(trajectory)
    options = []
    default_rates = price_of(trajectory.logged_model, pricing)
    for candidate in model.candidates:
        candidate_rates = price_of(candidate, pricing)
        if not (
            all(a <= b for a, b in zip(candidate_rates, default_rates))
            and any(a < b for a, b in zip(candidate_rates, default_rates))
        ):
            continue
        if propensities.get(candidate, 0.0) < MIN_PROPENSITY:
            continue
        if savings_estimator is not None:
            fixed_savings, adjusted_savings = savings_estimator.predict(
                trajectory, candidate,
            )
        else:
            # Unit-test/fallback estimate uses only the input visible at call zero.
            billed = trajectory.calls[0].input_tokens + trajectory.calls[0].tools_tokens
            fixed_savings = billed * (
                price_of(trajectory.logged_model, pricing)[0]
                - price_of(candidate, pricing)[0]
            ) / 1e6
            adjusted_savings = fixed_savings
        if fixed_savings <= 0 or adjusted_savings <= 0:
            continue
        risk_delta = risks[candidate] - risks[trajectory.logged_model]
        if max_individual_risk is not None and risk_delta > max_individual_risk:
            continue
        value_per_risk = adjusted_savings / max(risk_delta, 0.001)
        options.append((value_per_risk, adjusted_savings, fixed_savings, candidate, risk_delta))
    if not options:
        return trajectory.logged_model, ["no-positive-supported-downgrade"], risks, metadata
    value_per_risk, adjusted_savings, fixed_savings, candidate, risk_delta = max(options)
    metadata = {
        "fixed_savings": fixed_savings, "adjusted_savings": adjusted_savings,
        "risk_delta": risk_delta, "value_per_risk": value_per_risk,
    }
    reasons = [
        "positive-predecision-fixed-savings",
        "positive-predecision-adjusted-savings",
        "propensity-supported",
    ]
    reasons.extend(
        name for name, value in trajectory.feature_values.items()
        if value and name.startswith("intent_")
    )
    return candidate, reasons, risks, metadata


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


def _routes_from_initial(
    trajectory: Trajectory, model: FamilyModel, pricing: Dict[str, Sequence[float]],
    initial: str, reasons: List[str], risks: Dict[str, float],
    proposal: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
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
                    target = min(supported, key=lambda candidate: price_of(candidate, pricing)[0])
                    # Only information through the current input is used. Savings
                    # banked on earlier cheap calls must cover the reset and current
                    # expensive call; otherwise the escalation cannot break even
                    # against having kept the configured default from call zero.
                    trial = adaptive + [target]
                    default_prefix = [trajectory.logged_model] * (index + 1)
                    trial_cost = route_cost_at_decision(trajectory, trial, pricing)
                    default_cost = route_cost_at_decision(
                        trajectory, default_prefix, pricing,
                    )
                    if trial_cost <= default_cost + 1e-12:
                        current = target
                        switched = True
                        adaptive_reasons[index] = ["one-way-escalation", "cache-break-even"] + flags
                    else:
                        adaptive_reasons[index] = ["cache-break-even-failed"] + flags
        adaptive.append(current)
    return {
        "certified": certified, "adaptive": adaptive, "initial_reasons": reasons,
        "adaptive_reasons": adaptive_reasons, "risks": risks,
        "proposal": proposal or {
            "fixed_savings": 0.0, "adjusted_savings": 0.0,
            "risk_delta": 0.0, "value_per_risk": 0.0,
        },
    }


def make_routes(trajectory: Trajectory, model: FamilyModel, pricing: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    """Frozen v1 proposal rule, retained as the required current-policy ablation."""
    initial, reasons, risks = choose_initial(trajectory, model, pricing)
    return _routes_from_initial(trajectory, model, pricing, initial, reasons, risks)


def make_cost_effective_routes(
    trajectory: Trajectory, model: FamilyModel, pricing: Dict[str, Sequence[float]],
    output_ratios: Any, max_individual_risk: Optional[float] = None,
    savings_estimator: Optional["PredecisionSavings"] = None,
) -> Dict[str, Any]:
    initial, reasons, risks, proposal = choose_cost_effective_initial(
        trajectory, model, pricing, output_ratios, max_individual_risk,
        savings_estimator,
    )
    return _routes_from_initial(
        trajectory, model, pricing, initial, reasons, risks, proposal,
    )


def route_cost(
    trajectory: Trajectory, route: Sequence[str],
    pricing: Dict[str, Sequence[float]], include_observed_final: bool = False,
    output_ratios: Optional[Any] = None,
) -> Tuple[float, float]:
    total = switch_penalty = 0.0
    for index, (call, selected) in enumerate(zip(trajectory.calls, route)):
        same = index > 0 and route[index - 1] == selected
        cacheable = min(call.cacheable_tokens, call.input_tokens + call.tools_tokens) if same else 0
        pu, pc, po = price_of(selected, pricing)
        billed = call.input_tokens + call.tools_tokens
        output_tokens = call.output_tokens if (include_observed_final or not call.is_final) else 0
        if output_ratios and selected != trajectory.logged_model:
            if hasattr(output_ratios, "factor"):
                factor = output_ratios.factor(
                    trajectory, selected, trajectory.logged_model,
                )
            else:
                denominator = max(output_ratios.get(trajectory.logged_model, 1.0), 0.1)
                factor = output_ratios.get(selected, 1.0) / denominator
            output_tokens = int(output_tokens * factor)
        total += ((billed - cacheable) * pu + cacheable * pc + output_tokens * po) / 1e6
        if index > 0 and route[index - 1] != selected:
            switch_penalty += call.cacheable_tokens * (pu - pc) / 1e6
    return total, switch_penalty


def route_cost_at_decision(
    trajectory: Trajectory, route_prefix: Sequence[str],
    pricing: Dict[str, Sequence[float]],
) -> float:
    """Cost visible immediately before the last call in ``route_prefix`` runs.

    Prior observable outputs are included; the current call's future output is not.
    This is used only for adaptive cache break-even decisions.
    """
    total = 0.0
    for index, (call, selected) in enumerate(zip(trajectory.calls, route_prefix)):
        same = index > 0 and route_prefix[index - 1] == selected
        cacheable = min(
            call.cacheable_tokens, call.input_tokens + call.tools_tokens,
        ) if same else 0
        pu, pc, po = price_of(selected, pricing)
        billed = call.input_tokens + call.tools_tokens
        output_tokens = call.output_tokens if index < len(route_prefix) - 1 else 0
        total += (
            (billed - cacheable) * pu + cacheable * pc + output_tokens * po
        ) / 1e6
    return total


def output_length_ratios(trajectories: Sequence[Trajectory]) -> Dict[str, float]:
    """Frozen v1 model-level sensitivity, retained for the current-policy row."""
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


def _output_context(trajectory: Trajectory) -> Tuple[str, str]:
    values = trajectory.feature_values
    intent = next((
        name[len("intent_"):] for name in (
            "intent_high_stakes", "intent_side_effect", "intent_code",
            "intent_message", "intent_read",
        ) if values[name]
    ), "other")
    size = next((
        name[len("size_"):] for name in (
            "size_xs", "size_s", "size_m", "size_l",
        ) if values[name]
    ), "unknown")
    return intent, size


@dataclass
class OutputLengthSensitivity:
    """Training-only, partially pooled observable-output length estimates.

    Cells use only call-zero intent/size. Sparse model×context means shrink first
    to their model mean and then to the family mean. The final output remains absent.
    """

    predictions: Dict[Tuple[str, str, str], float]
    model_predictions: Dict[str, float]
    family_predictions: Dict[str, float]

    def expected(self, trajectory: Trajectory, model: str) -> float:
        intent, size = _output_context(trajectory)
        return self.predictions.get(
            (model, intent, size),
            self.model_predictions.get(model, self.family_predictions.get(_family(model), 1.0)),
        )

    def factor(self, trajectory: Trajectory, selected: str, logged: str) -> float:
        denominator = max(self.expected(trajectory, logged), 0.1)
        return min(max(self.expected(trajectory, selected) / denominator, 0.5), 2.0)


def fit_output_length_sensitivity(
    trajectories: Sequence[Trajectory], model_prior: float = 12.0,
    cell_prior: float = 10.0,
) -> OutputLengthSensitivity:
    """Fit output-length sensitivity on observable, non-final outputs in train."""
    observations: List[Tuple[Trajectory, float]] = []
    for trajectory in trajectories:
        if trajectory.split != "train":
            continue
        values = [call.output_tokens for call in trajectory.calls if not call.is_final]
        if values:
            observations.append((trajectory, statistics.fmean(values)))
    family_values: Dict[str, List[float]] = {}
    model_values: Dict[str, List[float]] = {}
    cell_values: Dict[Tuple[str, str, str], List[float]] = {}
    for trajectory, value in observations:
        family_values.setdefault(trajectory.family, []).append(value)
        model_values.setdefault(trajectory.logged_model, []).append(value)
        intent, size = _output_context(trajectory)
        cell_values.setdefault((trajectory.logged_model, intent, size), []).append(value)
    family_predictions = {
        family: statistics.fmean(values) for family, values in family_values.items()
    }
    model_predictions = {}
    for model, values in model_values.items():
        family_mean = family_predictions[_family(model)]
        model_predictions[model] = (
            sum(values) + model_prior * family_mean
        ) / (len(values) + model_prior)
    predictions = {}
    for key, values in cell_values.items():
        model_mean = model_predictions[key[0]]
        predictions[key] = (
            sum(values) + cell_prior * model_mean
        ) / (len(values) + cell_prior)
    return OutputLengthSensitivity(predictions, model_predictions, family_predictions)


def _savings_context(trajectory: Trajectory) -> Tuple[str, ...]:
    intent, size = _output_context(trajectory)
    values = trajectory.feature_values
    trigger = (
        "cron" if values["trigger_cron"] else
        "user" if values["trigger_user"] else
        "automation" if values["trigger_automation"] else "unknown"
    )
    surface = (
        "slack" if values["surface_slack"] else
        "teams" if values["surface_teams"] else "other"
    )
    return intent, size, trigger, surface, str(int(values["has_image"]))


@dataclass
class PredecisionSavings:
    """Training-only partially pooled savings labels, queried with call-zero data."""

    transition_means: Dict[Tuple[str, str], Tuple[float, float]]
    cell_values: Dict[Tuple[str, ...], Tuple[int, float, float]]
    prior_n: float = 8.0

    def predict(self, trajectory: Trajectory, candidate: str) -> Tuple[float, float]:
        transition = (trajectory.logged_model, candidate)
        fixed_prior, adjusted_prior = self.transition_means.get(transition, (0.0, 0.0))
        key = transition + _savings_context(trajectory)
        count, fixed_sum, adjusted_sum = self.cell_values.get(key, (0, 0.0, 0.0))
        return (
            (fixed_sum + self.prior_n * fixed_prior) / (count + self.prior_n),
            (adjusted_sum + self.prior_n * adjusted_prior) / (count + self.prior_n),
        )


def fit_predecision_savings(
    trajectories: Sequence[Trajectory], fitted: Dict[str, FamilyModel],
    pricing: Dict[str, Sequence[float]], output_sensitivity: OutputLengthSensitivity,
) -> PredecisionSavings:
    """Learn future cost shape from train; expose only opening-conditioned means."""
    transitions: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    cells: Dict[Tuple[str, ...], List[Tuple[float, float]]] = {}
    for trajectory in trajectories:
        if trajectory.split != "train":
            continue
        model = fitted[trajectory.family]
        if trajectory.logged_model not in model.candidates:
            continue
        logged_route = [trajectory.logged_model] * len(trajectory.calls)
        logged_fixed = route_cost(trajectory, logged_route, pricing)[0]
        logged_adjusted = route_cost(
            trajectory, logged_route, pricing, output_ratios=output_sensitivity,
        )[0]
        for candidate in model.candidates:
            candidate_rates = price_of(candidate, pricing)
            default_rates = price_of(trajectory.logged_model, pricing)
            if not (
                all(a <= b for a, b in zip(candidate_rates, default_rates))
                and any(a < b for a, b in zip(candidate_rates, default_rates))
            ):
                continue
            route = [candidate] * len(trajectory.calls)
            fixed = logged_fixed - route_cost(trajectory, route, pricing)[0]
            adjusted = logged_adjusted - route_cost(
                trajectory, route, pricing, output_ratios=output_sensitivity,
            )[0]
            transition = (trajectory.logged_model, candidate)
            transitions.setdefault(transition, []).append((fixed, adjusted))
            cells.setdefault(transition + _savings_context(trajectory), []).append(
                (fixed, adjusted)
            )
    transition_means = {
        key: (
            statistics.fmean(value[0] for value in values),
            statistics.fmean(value[1] for value in values),
        ) for key, values in transitions.items()
    }
    cell_values = {
        key: (
            len(values), sum(value[0] for value in values),
            sum(value[1] for value in values),
        ) for key, values in cells.items()
    }
    return PredecisionSavings(transition_means, cell_values)


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
    target_propensities: List[float] = []
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
            target_propensities.append(propensities[action])
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
        "routed_n": covered,
        "min_target_propensity": min(target_propensities) if target_propensities else 1.0,
        "median_target_propensity": statistics.median(target_propensities) if target_propensities else 1.0,
    }


def _proposal_priority(
    trajectory: Trajectory, route_data: Dict[str, Any],
    pricing: Dict[str, Sequence[float]], rank_by_value: bool = False,
) -> Tuple[Any, ...]:
    if rank_by_value:
        proposal = route_data["proposal"]
        return (
            -proposal["value_per_risk"], -proposal["adjusted_savings"],
            trajectory.trajectory_id,
        )
    initial = route_data["certified"][0]
    risks = route_data["risks"]
    risk_delta = risks[initial] - risks.get(trajectory.logged_model, risks[initial])
    return risk_delta, trajectory.trajectory_id


def _adopt_actions(
    trajectories: Sequence[Trajectory], proposals: Dict[str, Dict[str, Any]],
    pricing: Dict[str, Sequence[float]], fraction: float,
    rank_by_value: bool = False, value_threshold: Optional[float] = None,
) -> Dict[str, str]:
    changed = [t for t in trajectories if proposals[t.trajectory_id]["certified"][0] != t.logged_model]
    if value_threshold is not None:
        adopted = {
            t.trajectory_id for t in changed
            if proposals[t.trajectory_id]["proposal"]["value_per_risk"] >= value_threshold
        }
        return {
            t.trajectory_id: proposals[t.trajectory_id]["certified"][0]
            if t.trajectory_id in adopted else t.logged_model
            for t in trajectories
        }
    changed.sort(key=lambda t: _proposal_priority(
        t, proposals[t.trajectory_id], pricing, rank_by_value,
    ))
    take = int(round(fraction * len(changed)))
    adopted = {t.trajectory_id for t in changed[:take]}
    return {
        t.trajectory_id: proposals[t.trajectory_id]["certified"][0] if t.trajectory_id in adopted else t.logged_model
        for t in trajectories
    }


def _value_threshold(
    trajectories: Sequence[Trajectory], proposals: Dict[str, Dict[str, Any]],
    fraction: float,
) -> float:
    scores = sorted((
        proposals[t.trajectory_id]["proposal"]["value_per_risk"]
        for t in trajectories
        if proposals[t.trajectory_id]["certified"][0] != t.logged_model
    ), reverse=True)
    take = int(round(fraction * len(scores)))
    if take <= 0:
        return math.inf
    if take >= len(scores):
        return -math.inf
    # Continuous dollar/risk scores make ties rare. Include the kth proposal;
    # deterministic trajectory ids break any residual tie in the audit ledger.
    return scores[take - 1]


def _frontier(
    trajectories: Sequence[Trajectory], primary_fitted: Dict[str, FamilyModel],
    observed_fitted: Dict[str, FamilyModel], proposals: Dict[str, Dict[str, Any]],
    pricing: Dict[str, Sequence[float]], output_ratios: Any,
    selected_fraction: Optional[float] = None, rank_by_value: bool = False,
    fractions: Optional[Sequence[float]] = None, seed_prefix: str = "",
    value_thresholds: Optional[Dict[float, float]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Evaluate a frontier; select only when ``selected_fraction`` is absent.

    Selection minimizes behavior-adjusted cost under fixed/adjusted positive savings,
    the official (final-output-unobservable) 2pp upper bound, and routed ESS >= 15.
    """
    rows = []
    fractions = fractions or tuple(step / 20.0 for step in range(21))
    for fraction in fractions:
        actions = _adopt_actions(
            trajectories, proposals, pricing, fraction, rank_by_value,
            None if value_thresholds is None else value_thresholds[fraction],
        )
        primary = estimate_policy(
            trajectories, primary_fitted, actions, official=True,
            seed_suffix="%s-primary-%s" % (seed_prefix, fraction),
        )
        observed = estimate_policy(
            trajectories, observed_fitted, actions, official=False,
            seed_suffix="%s-observed-%s" % (seed_prefix, fraction),
        )
        logged_cost = routed_cost = adjusted_cost = logged_adjusted_cost = 0.0
        for trajectory in trajectories:
            logged_route = [trajectory.logged_model] * len(trajectory.calls)
            route = [actions[trajectory.trajectory_id]] * len(trajectory.calls)
            logged_cost += route_cost(trajectory, logged_route, pricing)[0]
            routed_cost += route_cost(trajectory, route, pricing)[0]
            adjusted_cost += route_cost(trajectory, route, pricing, output_ratios=output_ratios)[0]
            logged_adjusted_cost += route_cost(
                trajectory, logged_route, pricing, output_ratios=output_ratios,
            )[0]
        rows.append({
            "adopt_frac": fraction,
            "cost_usd_est": round(routed_cost, 4),
            "behavior_adjusted_cost_usd_est": round(adjusted_cost, 4),
            "savings_pct": round((1.0 - routed_cost / logged_cost) * 100.0, 2) if logged_cost else 0.0,
            "behavior_adjusted_savings_pct": round(
                (1.0 - adjusted_cost / logged_adjusted_cost) * 100.0, 2,
            ) if logged_adjusted_cost else 0.0,
            "proxy_quality": round(1.0 - primary["failure_delta"], 4),
            "failure_delta": round(primary["failure_delta"], 5),
            "failure_upper95": round(primary["failure_upper95"], 5),
            "observed_terminal_failure_delta": round(observed["failure_delta"], 5),
            "observed_terminal_failure_upper95": round(observed["failure_upper95"], 5),
            "coverage": round(primary["routed_coverage"], 4),
            "effective_n": round(primary["effective_n"], 1),
            "routed_n": primary["routed_n"],
            "min_target_propensity": round(primary["min_target_propensity"], 4),
            "median_target_propensity": round(primary["median_target_propensity"], 4),
            "selected": False,
        })
    if selected_fraction is None:
        eligible = [
            row for row in rows
            if row["failure_upper95"] <= QUALITY_MARGIN
            and row["savings_pct"] >= 0
            and row["behavior_adjusted_savings_pct"] >= 0
            and (row["routed_n"] == 0 or row["effective_n"] >= MIN_EFFECTIVE_N)
        ]
        selected = min(
            eligible, key=lambda row: row["behavior_adjusted_cost_usd_est"],
        ) if eligible else rows[0]
        selected_fraction = selected["adopt_frac"]
    selected = min(rows, key=lambda row: abs(row["adopt_frac"] - selected_fraction))
    selected["selected"] = True
    return rows, selected["adopt_frac"]


def corrected_baseline(
    trajectory: Trajectory, model: Optional[FamilyModel] = None,
    pricing: Optional[Dict[str, Sequence[float]]] = None,
) -> List[str]:
    """Call-zero prompt-size baseline with the same family/support guardrails."""
    initial = trajectory.calls[0].input_tokens + trajectory.calls[0].tools_tokens
    if initial >= 15_000 or model is None or trajectory.logged_model not in model.candidates:
        return [trajectory.logged_model] * len(trajectory.calls)
    pricing = pricing or load_pricing()
    default_rates = price_of(trajectory.logged_model, pricing)
    eligible = [
        candidate for candidate in model.candidates
        if all(a <= b for a, b in zip(price_of(candidate, pricing), default_rates))
        and model.propensity(trajectory).get(candidate, 0.0) >= MIN_PROPENSITY
    ]
    cheap = min(
        eligible, key=lambda candidate: price_of(candidate, pricing)[0],
        default=trajectory.logged_model,
    )
    return [cheap] * len(trajectory.calls)


def _write_csv(
    path: Path, rows: Sequence[Dict[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    fieldnames = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    current = summary["current_validation_point"]
    claim = (
        "Public validation: $%.2f → $%.2f, with upper95 %+.2f pp"
        % (current["cost_usd_est"], selected["cost_usd_est"], selected["failure_upper95"] * 100)
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
<section class="slide"><div class="kicker">The signal</div><h2>Global risk budget, local positive savings</h2><div class="cols"><div class="card"><h3>Nameable structure</h3><p>Tool dialect locks the family. At call zero we use trigger, intent, prompt size, image and toolset—never eventual depth, realized tools, future errors or the ending.</p></div><div class="card"><h3>The change</h3><p>Rank supported downgrades by <b>training-predicted cache-aware dollars / proxy risk</b>. Admit only actions predicted to save under fixed and output-length-adjusted costing. Keep one model for the trajectory.</p></div></div></section>
<section class="slide"><div class="kicker">The one chart</div><h2>Cost–proxy-quality frontier <span style="font-size:.5em">(public validation; held-out pending)</span></h2><img class="chart" src="results/routemaxxing/frontier.svg" alt="frontier"></section>
<section class="slide"><div class="kicker">Honesty</div><h2>What the log can—and cannot—identify</h2><ul style="list-style:none"><li><b style="color:var(--violet)">Method —</b> Family-restricted propensity screen + outcome regression; 2,000 trajectory bootstraps. Primary proxy excludes the unobservable final answer.</li><li style="margin-top:2vh"><b style="color:var(--violet)">Support —</b> Routed ESS %.1f; minimum target propensity %.3f. Luna has 12 training rows versus 20 required, so it is never selected.</li><li style="margin-top:2vh"><b style="color:var(--violet)">Known weakness —</b> Proxy risk is not measured task quality, and a different early model may create a different later state. Adaptive results are conditional, not causal.</li></ul></section>
<section class="slide dark"><div class="kicker">One claim</div><h1 style="font-size:3.5vw">The old “within 2pp of best model” gate left safe dollars behind.</h1><p class="big">%.1f%% → %.1f%% estimated savings · %.1f%% routed coverage</p><p>Lock now. Run once on organizer held-out. Do not retune.</p></section>
<div class="foot">routemaxxing · Viktor challenge · TUM.ai</div><script>const s=[...document.querySelectorAll('.slide')];let i=0;const show=n=>{s[i].classList.remove('active');i=(n+s.length)%%s.length;s[i].classList.add('active')};addEventListener('keydown',e=>{if(['ArrowRight',' ','PageDown'].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key==='f')document.documentElement.requestFullscreen()});addEventListener('click',()=>show(i+1));</script></body></html>""" % (
        html.escape(claim), selected["effective_n"], selected["min_target_propensity"],
        current["savings_pct"], selected["savings_pct"], selected["coverage"] * 100,
    )
    path.write_text(document, encoding="utf-8")


def run(export: str = "export", results_root: Path = ROOT / "results") -> Dict[str, Any]:
    pricing = load_pricing()
    trajectories = load_export(export)
    train = [t for t in trajectories if t.split == "train"]
    validation = [t for t in trajectories if t.split == "validation"]

    # The v1 models reproduce the current certified router. New policy candidates
    # use the final-output-unobservable proxy for decisions and evaluation.
    v1_observed = fit_models(trajectories, official=False)
    v1_primary = fit_models(trajectories, official=True)
    legacy_ratios = output_length_ratios(trajectories)
    output_sensitivity = fit_output_length_sensitivity(trajectories)
    predecision_savings = fit_predecision_savings(
        trajectories, v1_primary, pricing, output_sensitivity,
    )
    current_proposals = {
        t.trajectory_id: make_routes(t, v1_observed[t.family], pricing)
        for t in trajectories
    }
    current_train_frontier, current_fraction = _frontier(
        train, v1_primary, v1_observed, current_proposals, pricing,
        legacy_ratios, rank_by_value=False, seed_prefix="current-train",
    )
    current_train_selected = next(row for row in current_train_frontier if row["selected"])

    specifications = [
        {
            "name": "value_rank_v1_features",
            "description": "value-per-risk ranking with frozen v1 opening features",
            "primary": v1_primary,
            "observed": v1_observed,
        },
        {
            "name": "enriched_opening_features",
            "description": "adds call-zero trigger, size, toolset, surface and side-effect indicators",
            "primary": fit_models(
                trajectories, official=True, feature_names=ENRICHED_FEATURE_NAMES,
                outcome_l2=4.0, propensity_l2=4.0,
            ),
            "observed": fit_models(
                trajectories, official=False, feature_names=ENRICHED_FEATURE_NAMES,
                outcome_l2=4.0, propensity_l2=4.0,
            ),
        },
        {
            "name": "partial_pooling_ablation",
            "description": "enriched features with fixed stronger regularization and propensity shrinkage",
            "primary": fit_models(
                trajectories, official=True, feature_names=ENRICHED_FEATURE_NAMES,
                outcome_l2=8.0, propensity_l2=8.0, propensity_alpha=20.0,
            ),
            "observed": fit_models(
                trajectories, official=False, feature_names=ENRICHED_FEATURE_NAMES,
                outcome_l2=8.0, propensity_l2=8.0, propensity_alpha=20.0,
            ),
        },
    ]

    # Every knob below is swept on train only. Validation is not evaluated until
    # ``locked`` is assigned. None means no per-trajectory cap; the global 2pp UCB
    # and ESS constraints still apply.
    risk_caps: Tuple[Optional[float], ...] = (0.0, 0.01, 0.02, None)
    search_fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    training_sweep: List[Dict[str, Any]] = []
    quality_margin_sweep: List[Dict[str, Any]] = []
    best_by_spec: Dict[str, Dict[str, Any]] = {}
    for specification in specifications:
        for cap in risk_caps:
            candidate_proposals = {
                t.trajectory_id: make_cost_effective_routes(
                    t, specification["primary"][t.family], pricing,
                    output_sensitivity, cap, predecision_savings,
                ) for t in trajectories
            }
            rows, selected_fraction = _frontier(
                train, specification["primary"], specification["observed"],
                candidate_proposals, pricing, output_sensitivity,
                rank_by_value=True, fractions=search_fractions,
                seed_prefix="%s-cap-%s-train" % (specification["name"], cap),
            )
            for row in rows:
                training_sweep.append({
                    "policy": specification["name"],
                    "risk_cap": "none" if cap is None else cap,
                    **row,
                })
            selected = next(row for row in rows if row["selected"])
            record = {
                **specification, "risk_cap": cap,
                "proposals": candidate_proposals,
                "train_frontier": rows, "selected_fraction": selected_fraction,
                "train_selected": selected,
            }
            previous = best_by_spec.get(specification["name"])
            if previous is None or (
                selected["behavior_adjusted_cost_usd_est"], selected["cost_usd_est"]
            ) < (
                previous["train_selected"]["behavior_adjusted_cost_usd_est"],
                previous["train_selected"]["cost_usd_est"],
            ):
                best_by_spec[specification["name"]] = record

    # Reuse the same training bootstrap points to show how the locked solution
    # changes under tighter global UCB margins; no validation statistic enters.
    for specification in specifications:
        for cap in risk_caps:
            candidates = [
                row for row in training_sweep
                if row["policy"] == specification["name"]
                and row["risk_cap"] == ("none" if cap is None else cap)
            ]
            for margin in (0.0, 0.005, 0.01, QUALITY_MARGIN):
                eligible = [
                    row for row in candidates
                    if row["failure_upper95"] <= margin
                    and row["savings_pct"] >= 0
                    and row["behavior_adjusted_savings_pct"] >= 0
                    and (row["routed_n"] == 0 or row["effective_n"] >= MIN_EFFECTIVE_N)
                ]
                selected_for_margin = min(
                    eligible,
                    key=lambda row: row["behavior_adjusted_cost_usd_est"],
                ) if eligible else candidates[0]
                quality_margin_sweep.append({
                    "policy": specification["name"],
                    "individual_risk_cap": "none" if cap is None else cap,
                    "global_upper95_margin": margin,
                    **{key: value for key, value in selected_for_margin.items()
                       if key not in ("policy", "risk_cap")},
                })

    # Policy lock: the winner is determined only by constrained training cost.
    eligible_candidates = [
        record for record in best_by_spec.values()
        if record["train_selected"]["failure_upper95"] <= QUALITY_MARGIN
        and record["train_selected"]["behavior_adjusted_cost_usd_est"]
        < current_train_selected["behavior_adjusted_cost_usd_est"]
        and record["train_selected"]["cost_usd_est"]
        < current_train_selected["cost_usd_est"]
    ]
    locked = min(
        eligible_candidates,
        key=lambda record: (
            record["train_selected"]["behavior_adjusted_cost_usd_est"],
            record["train_selected"]["cost_usd_est"],
        ),
    ) if eligible_candidates else None
    if locked is None:
        final_name = "current_routemaxxing_certified"
        final_primary, final_observed = v1_primary, v1_observed
        proposals = current_proposals
        selected_fraction = current_fraction
        final_threshold = None
        rank_by_value = False
        final_risk_cap = None
    else:
        final_name = locked["name"]
        final_primary, final_observed = locked["primary"], locked["observed"]
        proposals = locked["proposals"]
        selected_fraction = locked["selected_fraction"]
        final_threshold = _value_threshold(train, proposals, selected_fraction)
        rank_by_value = True
        final_risk_cap = locked["risk_cap"]

    # Public validation starts here, after the complete policy lock above.
    current_frontier, _ = _frontier(
        validation, v1_primary, v1_observed, current_proposals, pricing,
        legacy_ratios, current_fraction, rank_by_value=False,
        seed_prefix="current-validation",
    )
    comparison: List[Dict[str, Any]] = []
    comparison.append({
        "phase": "validation", "policy": "logged_behavior",
        **next(row for row in current_frontier if row["adopt_frac"] == 0.0),
    })
    current_validation_selected = next(row for row in current_frontier if row["selected"])
    comparison.append({
        "phase": "validation", "policy": "current_routemaxxing_certified",
        **current_validation_selected,
    })
    current_repriced_frontier, _ = _frontier(
        validation, v1_primary, v1_observed, current_proposals, pricing,
        output_sensitivity, current_fraction, rank_by_value=False,
        fractions=(current_fraction,), seed_prefix="current-repriced-validation",
    )
    comparison.append({
        "phase": "validation",
        "policy": "current_router_output_sensitivity_ablation",
        "note": "same routes; partially pooled output-length sensitivity",
        **current_repriced_frontier[0],
    })

    validation_by_spec: Dict[str, Dict[str, Any]] = {}
    for name, record in best_by_spec.items():
        fraction = record["selected_fraction"]
        threshold = _value_threshold(train, record["proposals"], fraction)
        rows, _ = _frontier(
            validation, record["primary"], record["observed"],
            record["proposals"], pricing, output_sensitivity, fraction,
            rank_by_value=True, fractions=(fraction,),
            value_thresholds={fraction: threshold},
            seed_prefix="%s-validation" % name,
        )
        validation_by_spec[name] = rows[0]
        comparison.append({
            "phase": "validation", "policy": name,
            "risk_cap": "none" if record["risk_cap"] is None else record["risk_cap"],
            **rows[0],
        })

    if locked is None:
        frontier = current_frontier
    else:
        frontier_fractions = tuple(step / 20.0 for step in range(21))
        thresholds = {
            fraction: _value_threshold(train, proposals, fraction)
            for fraction in frontier_fractions
        }
        frontier, _ = _frontier(
            validation, final_primary, final_observed, proposals, pricing,
            output_sensitivity, selected_fraction, rank_by_value=True,
            fractions=frontier_fractions, value_thresholds=thresholds,
            seed_prefix="final-validation-frontier",
        )

    outdir = Path(results_root) / "routemaxxing"
    outdir.mkdir(parents=True, exist_ok=True)
    selected_actions = _adopt_actions(
        trajectories, proposals, pricing, selected_fraction, rank_by_value,
        final_threshold,
    )
    route_records = []
    adaptive_records = []
    ledger = []
    totals = Counter()
    for trajectory in trajectories:
        proposal = proposals[trajectory.trajectory_id]
        chosen = selected_actions[trajectory.trajectory_id]
        certified = [chosen] * len(trajectory.calls)
        if _family(chosen) != trajectory.family:
            raise AssertionError("certified cross-family route: %s" % trajectory.trajectory_id)
        if any(
            candidate_rate > default_rate
            for candidate_rate, default_rate in zip(
                price_of(chosen, pricing), price_of(trajectory.logged_model, pricing),
            )
        ):
            raise AssertionError("certified upgrade: %s" % trajectory.trajectory_id)
        adaptive = proposal["adaptive"] if chosen != trajectory.logged_model else certified
        logged_route = [trajectory.logged_model] * len(trajectory.calls)
        logged_cost, _ = route_cost(trajectory, logged_route, pricing)
        certified_cost, certified_penalty = route_cost(trajectory, certified, pricing)
        if certified_cost > logged_cost + 1e-12:
            raise AssertionError("negative certified savings: %s" % trajectory.trajectory_id)
        adaptive_cost, adaptive_penalty = route_cost(trajectory, adaptive, pricing)
        adjusted_cost, _ = route_cost(
            trajectory, certified, pricing, output_ratios=output_sensitivity,
        )
        if adjusted_cost > logged_cost + 1e-12:
            raise AssertionError(
                "negative behavior-adjusted certified savings: %s"
                % trajectory.trajectory_id
            )
        # A conditional escalation is never emitted if the completed logged-state
        # route would erase all fixed-shape savings.
        if adaptive_cost > logged_cost + 1e-12:
            adaptive = certified
            adaptive_cost, adaptive_penalty = certified_cost, 0.0
        switches = sum(adaptive[i] != adaptive[i - 1] for i in range(1, len(adaptive)))
        if switches > 1 or any(
            price_of(adaptive[i], pricing)[0] < price_of(adaptive[i - 1], pricing)[0]
            for i in range(1, len(adaptive)) if adaptive[i] != adaptive[i - 1]
        ):
            raise AssertionError("adaptive route is not one-way: %s" % trajectory.trajectory_id)
        baseline = corrected_baseline(
            trajectory, v1_primary[trajectory.family], pricing,
        )
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
            "proposal": proposal["proposal"],
        })
        adaptive_records.append({
            **common, "route": adaptive,
            "cost_logged_usd": round(logged_cost, 8),
            "cost_routed_usd": round(adaptive_cost, 8),
            "switches": switches,
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
                "propensity_by_model": final_primary[trajectory.family].propensity(trajectory),
                "in_support": final_primary[trajectory.family].propensity(trajectory).get(selected, 0) >= MIN_PROPENSITY,
                "policy": final_name,
                "proposal": proposal["proposal"],
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
    train_frontier = locked["train_frontier"] if locked is not None else current_train_frontier
    _write_csv(outdir / "train_frontier.csv", train_frontier)
    _write_csv(outdir / "training_sweep.csv", training_sweep)
    _write_csv(outdir / "quality_margin_sweep.csv", quality_margin_sweep)
    _write_csv(outdir / "frontier.csv", frontier)
    _frontier_svg(outdir / "frontier.svg", frontier)

    selected = next(row for row in frontier if row["selected"])
    final_train_selected = (
        locked["train_selected"] if locked is not None else current_train_selected
    )

    # Required heuristic comparison uses call-zero prompt size and the same support
    # guards; it is not allowed to peek at eventual trajectory depth.
    baseline_actions = {
        t.trajectory_id: corrected_baseline(t, v1_primary[t.family], pricing)[0]
        for t in validation
    }
    baseline_primary = estimate_policy(
        validation, v1_primary, baseline_actions, official=True,
        seed_suffix="corrected-baseline-primary",
    )
    baseline_observed = estimate_policy(
        validation, v1_observed, baseline_actions, official=False,
        seed_suffix="corrected-baseline-observed",
    )
    validation_logged_cost = validation_baseline_cost = validation_baseline_adjusted = 0.0
    for trajectory in validation:
        logged_route = [trajectory.logged_model] * len(trajectory.calls)
        baseline_route = corrected_baseline(
            trajectory, v1_primary[trajectory.family], pricing,
        )
        validation_logged_cost += route_cost(trajectory, logged_route, pricing)[0]
        validation_baseline_cost += route_cost(trajectory, baseline_route, pricing)[0]
        validation_baseline_adjusted += route_cost(
            trajectory, baseline_route, pricing, output_ratios=output_sensitivity,
        )[0]
    comparison.insert(1, {
        "phase": "validation", "policy": "corrected_heuristic_baseline",
        "adopt_frac": 1.0,
        "cost_usd_est": round(validation_baseline_cost, 4),
        "behavior_adjusted_cost_usd_est": round(validation_baseline_adjusted, 4),
        "savings_pct": round((1.0 - validation_baseline_cost / validation_logged_cost) * 100, 2),
        "behavior_adjusted_savings_pct": round((1.0 - validation_baseline_adjusted / validation_logged_cost) * 100, 2),
        "failure_delta": round(baseline_primary["failure_delta"], 5),
        "failure_upper95": round(baseline_primary["failure_upper95"], 5),
        "observed_terminal_failure_delta": round(baseline_observed["failure_delta"], 5),
        "observed_terminal_failure_upper95": round(baseline_observed["failure_upper95"], 5),
        "coverage": round(baseline_primary["routed_coverage"], 4),
        "effective_n": round(baseline_primary["effective_n"], 1),
        "routed_n": baseline_primary["routed_n"],
        "min_target_propensity": round(baseline_primary["min_target_propensity"], 4),
        "median_target_propensity": round(baseline_primary["median_target_propensity"], 4),
        "note": "call-zero prompt-size heuristic; family/support guarded",
    })
    comparison.append({
        "phase": "validation", "policy": "final_improved_certified",
        "implementation": final_name,
        "risk_cap": "none" if final_risk_cap is None else final_risk_cap,
        **selected,
    })
    validation_adaptive = [row for row in adaptive_records if row["split"] == "validation"]
    adaptive_validation_cost = sum(row["cost_routed_usd"] for row in validation_adaptive)
    adaptive_switches = sum(row["switches"] for row in validation_adaptive)
    comparison.append({
        "phase": "validation", "policy": "adaptive_conditional",
        "cost_usd_est": round(adaptive_validation_cost, 4),
        "savings_pct": round((1.0 - adaptive_validation_cost / validation_logged_cost) * 100, 2),
        "coverage": round(sum(row["route"][0] != row["logged_model"] for row in validation_adaptive) / len(validation_adaptive), 4),
        "switches": adaptive_switches,
        "note": "conditional on logged states; no end-to-end failure estimate",
    })
    _write_csv(outdir / "ablation_comparison.csv", comparison)

    proposal_rankings = []
    expensive_defaults = []
    for trajectory in trajectories:
        proposal = proposals[trajectory.trajectory_id]
        proposed = proposal["certified"][0]
        chosen = selected_actions[trajectory.trajectory_id]
        if proposed != trajectory.logged_model:
            proposal_rankings.append({
                "trajectory_id": trajectory.trajectory_id,
                "split": trajectory.split, "logged_model": trajectory.logged_model,
                "proposed_model": proposed, "adopted": chosen == proposed,
                **proposal["proposal"],
            })
        supported = trajectory.logged_model in final_primary[trajectory.family].candidates
        has_cheaper = any(
            price_of(candidate, pricing)[0] < price_of(trajectory.logged_model, pricing)[0]
            for candidate in final_primary[trajectory.family].candidates
        )
        if supported and has_cheaper and chosen == trajectory.logged_model:
            reason = (
                "below-training-value-threshold" if proposed != trajectory.logged_model
                else ";".join(proposal["initial_reasons"])
            )
            expensive_defaults.append({
                "trajectory_id": trajectory.trajectory_id, "split": trajectory.split,
                "logged_model": trajectory.logged_model, "reason": reason,
                "best_proposed_model": proposed, **proposal["proposal"],
            })
    if proposal_rankings:
        proposal_rankings.sort(key=lambda row: row["value_per_risk"], reverse=True)
        _write_csv(outdir / "proposal_rankings.csv", proposal_rankings)
    _write_csv(
        outdir / "expensive_defaults.csv", expensive_defaults,
        (
            "trajectory_id", "split", "logged_model", "reason",
            "best_proposed_model", "fixed_savings", "adjusted_savings",
            "risk_delta", "value_per_risk",
        ),
    )

    support = {}
    for family, model in final_primary.items():
        support[family] = {
            candidate: {
                "train_n": model.counts.get(candidate, 0),
                "eligible": candidate in model.candidates,
                "minimum_n": MIN_MODEL_OBS,
                "reason": "eligible" if candidate in model.candidates else "below minimum training support",
            } for candidate in FAMILY_CANDIDATES[family]
        }
    (outdir / "propensity_support.json").write_text(
        json.dumps(support, indent=2) + "\n", encoding="utf-8",
    )
    output_report = {
        "method": "training-only partial pooling; observable non-final output tokens only",
        "model_expected_output_tokens_est": output_sensitivity.model_predictions,
        "family_expected_output_tokens_est": output_sensitivity.family_predictions,
        "context_cells": [
            {
                "model": key[0], "intent": key[1], "initial_size": key[2],
                "expected_output_tokens_est": value,
            }
            for key, value in sorted(output_sensitivity.predictions.items())
        ],
    }
    (outdir / "output_length_sensitivity.json").write_text(
        json.dumps(output_report, indent=2) + "\n", encoding="utf-8",
    )

    logged = totals["logged_micros"] / 1e6
    summary = {
        "team": "routemaxxing",
        "members": ["Atahan Kılıç", "Musa Kocabasoğlu", "Utku Kalkanlı"],
        "data_interpretation": "one complete history per row; final output excluded in primary cost and primary proxy",
        "trajectories": len(trajectories), "calls": sum(len(t.calls) for t in trajectories),
        "train_trajectories": len(train), "validation_trajectories": len(validation),
        "estimated_tokens": "serialized JSON characters / 4; not measured usage",
        "logged_cost_usd_est": round(logged, 4),
        "corrected_baseline_cost_usd_est": round(totals["baseline_micros"] / 1e6, 4),
        "certified_cost_usd_est": round(totals["certified_micros"] / 1e6, 4),
        "adaptive_cost_usd_est": round(totals["adaptive_micros"] / 1e6, 4),
        "behavior_adjusted_certified_cost_usd_est": round(totals["adjusted_micros"] / 1e6, 4),
        "locked_policy": final_name,
        "locked_risk_cap": final_risk_cap,
        "training_value_threshold": None if final_threshold is None or not math.isfinite(final_threshold) else final_threshold,
        "training_adoption_rule": (
            "all positive supported proposals" if selected_fraction == 1.0
            else "value-per-risk score at or above locked training threshold"
        ),
        "predecision_cost_model": "training-only partially pooled transition savings by opening intent, initial size, trigger, surface, and image",
        "future_information_excluded": "eventual tools, final depth, future errors, observable outputs, and final answer",
        "supported_expensive_defaults_remaining": len(expensive_defaults),
        "selected_adopt_frac": selected_fraction,
        "training_selected_point": final_train_selected,
        "validation_selected_point": selected,
        "current_validation_point": current_validation_selected,
        "current_validation_repriced_output_sensitivity": current_repriced_frontier[0],
        "validation_fixed_cost_improvement_vs_current_usd_est": round(
            current_validation_selected["cost_usd_est"] - selected["cost_usd_est"], 4,
        ),
        "validation_behavior_adjusted_improvement_vs_current_usd_est": round(
            current_repriced_frontier[0]["behavior_adjusted_cost_usd_est"]
            - selected["behavior_adjusted_cost_usd_est"], 4,
        ),
        "quality_margin": QUALITY_MARGIN,
        "bootstrap_replicates": BOOTSTRAPS,
        "held_out_status": "not present; policy locked on public train and reported on public validation",
        "adaptive_estimate": "conditional on logged states; not an end-to-end causal claim",
        "known_weakness": "automatic proxy labels and counterfactual state drift after an alternative early model",
        "primary_proxy": "unresolved final observable tool error or repeated identical tool call; terminal assistant output excluded",
        "sensitivity_proxy": "primary proxy plus terminal failure language from the observable row ending",
        "off_policy_method": "family-restricted, propensity-screened outcome regression for downgrade-from-default effects with 2,000 trajectory bootstraps",
        "policy_selection": "feature, risk-cap, and adoption sweeps on train only; value threshold locked before public validation",
        "luna_support": support["gpt"]["gpt-5.6-luna"],
        "artifacts": {
            "comparison": "ablation_comparison.csv",
            "training_sweep": "training_sweep.csv",
            "quality_margin_sweep": "quality_margin_sweep.csv",
            "ranked_proposals": "proposal_rankings.csv",
            "expensive_defaults": "expensive_defaults.csv",
            "support": "propensity_support.json",
            "output_length_sensitivity": "output_length_sensitivity.json",
        },
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
