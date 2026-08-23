#!/usr/bin/env python3
"""Versioned, near-oracle router with honest offline evaluation.

This file intentionally lives beside the starter routers and does not modify them.
It corrects a property of the real chunk that the starter baseline misses: one JSONL
row is one complete trajectory.  Model-produced runs inside the row are the actual
calls; intervening tool results are the next call's environment input.

The deployable policy uses only information present before the first model output.
For every supported model in the request's existing harness family it estimates:

* future uncached input, cached input, and output tokens;
* cache-aware input + output cost at the official rates;
* an execution outcome proxy and its uncertainty;
* effective matched support in the offline data.

Estimates come from transparent nearest-neighbour matching.  Evaluation is five-fold
cross-fitted.  A clipped doubly-robust (DR) estimate is reported for policies that do
not depend on the logged action.  The starter policy is also re-evaluated on the same
corrected ledger, but is labelled as a model-assisted estimate because it uses final
trajectory size and the logged model.

All token counts are serialized-character / 4 estimates, never measured usage.

Usage:
    python3.13 scripts/routers/v2_oracle.py export/

Writes only new ignored artifacts under results/router_v*/ and results/oracle_router/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cost_model import load_pricing, price_of  # noqa: E402
from decompose import decompose  # noqa: E402
from trajectory_ids import first_user_text  # noqa: E402


SUPPORTED = {
    "claude": ("claude-sonnet-5", "claude-opus-5", "claude-fable-5"),
    "gpt": ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"),
}

CHEAP = {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-luna"}
STRONG_DEFAULT = {"claude": "claude-opus-5", "gpt": "gpt-5.6-terra"}

STOPWORDS = {
    "about", "after", "again", "also", "and", "any", "are", "been", "before",
    "being", "but", "can", "could", "current", "did", "does", "each", "for",
    "from", "had", "has", "have", "here", "into", "its", "just", "more",
    "must", "not", "now", "only", "our", "out", "please", "run", "same",
    "should", "some", "than", "that", "the", "their", "then", "there", "these",
    "they", "this", "those", "through", "today", "use", "using", "was", "were",
    "what", "when", "where", "which", "while", "will", "with", "within", "would",
    "you", "your", "system", "auto", "read", "path", "true", "false", "none",
    "task", "thread", "info", "output", "message", "user", "assistant", "slack",
}

WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{2,}")
PII_RE = re.compile(r"^(?:pii|id|url|person|project|company|path|file)_", re.I)
EXIT_RE = re.compile(r"\b(?:exit code|exit_code)\s*[=:]\s*(-?\d+)", re.I)
START_ERROR_RE = re.compile(
    r"^(?:error|failed|exception|traceback|validation error|tool error)\s*[:\-]", re.I
)
FALSE_STATUS_RE = re.compile(r'"(?:success|ok)"\s*:\s*false', re.I)
NEGATIVE_FINAL_RE = re.compile(
    r"\b(?:i(?:'m| am)? sorry|unable to|could not|couldn't|can't complete|"
    r"cannot complete|blocked from|failed to complete)\b", re.I
)

TASK_TAGS = {
    "cron": ("cron", "scheduled", "monitor", "watcher", "poll"),
    "communication": ("slack", "message", "reply", "email", "teams", "notify"),
    "coding": ("code", "script", "debug", "implement", "patch", "repository"),
    "research": ("research", "search", "find", "investigate", "source", "audit"),
    "document": ("document", "report", "summary", "draft", "write", "canvas"),
    "data": ("csv", "json", "database", "query", "dashboard", "spreadsheet"),
    "media": ("audio", "image", "video", "attachment", "transcrib"),
    "external_action": ("send", "post", "create", "update", "delete", "submit"),
}

SCHEDULED_TASK_RE = re.compile(
    r"- Task \(from scheduler\):\s*(.*?)(?=\n- Cron path:|</system>|\Z)", re.I | re.S
)
MESSAGE_MARKERS = ("[Slack message", "[Microsoft Teams message")


@dataclass(frozen=True)
class Context:
    """Only fields known before the first model output."""

    row: int
    trajectory_id: str
    family: str
    requested_model: str
    task_key: str
    initial_tokens: int
    user_chars: int
    tools_tokens: int
    n_tools: int
    n_initial_items: int
    attachment_count: int
    numeric: tuple[float, ...]
    words: frozenset[str]
    tools: frozenset[str]
    tags: frozenset[str]


@dataclass(frozen=True)
class Trajectory:
    context: Context
    model: str
    n_calls: int
    uncached_tokens: int
    cached_tokens: int
    output_tokens: int
    full_input_tokens: int
    observed_cost: float
    outcome: float

    @property
    def future_uncached_ratio(self) -> float:
        return max(0, self.uncached_tokens - self.context.initial_tokens) / max(
            self.context.initial_tokens, 1
        )

    @property
    def cached_ratio(self) -> float:
        return self.cached_tokens / max(self.context.initial_tokens, 1)

    @property
    def output_ratio(self) -> float:
        return self.output_tokens / max(self.context.initial_tokens, 1)

    @property
    def total_tokens(self) -> int:
        return self.uncached_tokens + self.cached_tokens + self.output_tokens


@dataclass(frozen=True)
class Prediction:
    model: str
    uncached_tokens: float
    cached_tokens: float
    output_tokens: float
    cost_usd: float
    quality: float
    quality_se: float
    effective_neighbors: float
    nearest_distance: float
    support_radius: float = float("inf")
    analogue_neighbors: int = 0
    total_tokens_p50: float = 0.0


@dataclass(frozen=True)
class EvaluationRow:
    record: Trajectory
    predictions: dict[str, Prediction]
    propensities: dict[str, float]
    actions: dict[str, str]


def _json_tokens(obj) -> int:
    return len(json.dumps(obj)) // 4


def _item_tokens(items: Sequence[dict]) -> list[int]:
    return [len(json.dumps(item)) // 4 for item in items]


def _content_text(item: dict) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


def _tool_name(tool: dict) -> str:
    body = tool.get("function", tool)
    return str(body.get("name", "unknown"))


def _family(row: dict) -> str:
    # Tool schemas identify the immutable harness independently of the behavior
    # model.  Only use the requested model as a fallback for unfamiliar schemas.
    names = {_tool_name(t) for t in (row.get("tools") or [])}
    if {"shell_command", "apply_patch"} & names:
        return "gpt"
    if {"bash", "file_read", "file_edit", "file_write"} & names:
        return "claude"
    model = str(row.get("model", ""))
    if model.startswith("gpt"):
        return "gpt"
    if model.startswith("claude"):
        return "claude"
    return "claude"


def _initial_items(row: dict) -> tuple[list[dict], list[tuple[int, int]]]:
    items = row.get("input") or []
    spans = decompose(items)
    # A live first-call request has no model output yet, hence no spans.
    start = spans[0][0] if spans else len(items)
    return items[:start], spans


def _task_words(text: str, limit: int = 220) -> frozenset[str]:
    # Task instructions and the newest inbound message are normally at the tail.
    sample = (text[:1200] + " " + text[-8000:]).lower()
    counts = Counter(
        word for word in WORD_RE.findall(sample)
        if word not in STOPWORDS and not PII_RE.match(word) and not word.isdigit()
    )
    return frozenset(word for word, _ in counts.most_common(limit))


def _routing_task_text(text: str) -> str:
    """Isolate the actual scheduler task or newest platform message from memory noise."""
    scheduled = SCHEDULED_TASK_RE.search(text)
    if scheduled:
        return scheduled.group(1).strip()
    positions = [text.rfind(marker) for marker in MESSAGE_MARKERS]
    start = max(positions)
    return text[start:].strip() if start >= 0 else text[-2500:].strip()


def _normalized_task_key(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\bpii_[a-z0-9_]+\b", "<pii>", normalized)
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    normalized = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{20,}\b", "<uuid>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()[:12]


def context_from_row(row: dict, row_idx: int = -1) -> Context:
    """Extract routing features without looking past the first model output."""
    initial, _ = _initial_items(row)
    tools = row.get("tools") or []
    tools_tokens = _json_tokens(tools)
    initial_tokens = sum(_item_tokens(initial)) + tools_tokens
    text = " ".join(
        _content_text(item) for item in initial
        if item.get("role") in {"system", "user"}
    )
    # first_user_text is still useful for legacy message shapes.
    if not text:
        text = first_user_text(row)
    task_text = _routing_task_text(text)
    low = task_text.lower()
    tags = frozenset(
        tag for tag, needles in TASK_TAGS.items() if any(n in low for n in needles)
    )
    names = frozenset(_tool_name(t) for t in tools)
    attachment_count = len(re.findall(r"attachment|\.wav\b|\.mp3\b|\.png\b|\.pdf\b", low))
    numeric = (
        math.log1p(initial_tokens),
        math.log1p(len(text)),
        math.log1p(tools_tokens),
        float(len(tools)),
        float(len(initial)),
        math.log1p(attachment_count),
        float("cron" in tags),
        float("external_action" in tags),
    )
    digest = hashlib.sha1(json.dumps(initial, sort_keys=True).encode()).hexdigest()[:12]
    return Context(
        row=row_idx,
        trajectory_id=digest,
        family=_family(row),
        requested_model=str(row.get("model", "")),
        task_key=_normalized_task_key(task_text),
        initial_tokens=initial_tokens,
        user_chars=len(text),
        tools_tokens=tools_tokens,
        n_tools=len(tools),
        n_initial_items=len(initial),
        attachment_count=attachment_count,
        numeric=numeric,
        words=_task_words(task_text),
        tools=names,
        tags=tags,
    )


def _decoded_output(output) -> tuple[str, dict | None]:
    if not isinstance(output, str):
        return json.dumps(output), output if isinstance(output, dict) else None
    try:
        obj = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict):
        return str(obj.get("content", output)), obj
    return output, None


def tool_result_failed(output) -> bool:
    """High-precision execution failure detector; avoids generic 'error' matches."""
    text, obj = _decoded_output(output)
    if obj:
        exit_code = obj.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        if str(obj.get("status", "")).lower() in {
            "error", "failed", "failure", "timeout", "timed_out", "denied", "invalid"
        }:
            return True
        if obj.get("success") is False or obj.get("ok") is False:
            return True
    match = EXIT_RE.search(text[:400])
    if match and int(match.group(1)) != 0:
        return True
    stripped = text.strip()
    return bool(
        START_ERROR_RE.search(stripped)
        or FALSE_STATUS_RE.search(stripped[:1200])
        or "failed to validate tool" in stripped[:600].lower()
    )


def outcome_score(row: dict, spans: Sequence[tuple[int, int]] | None = None) -> float:
    """Execution-completion proxy in [0, 1], based only on logged outcomes.

    Failures recovered by a later successful call to the same tool receive a small
    efficiency penalty.  A tool whose last result is still failed, or a terminal
    response admitting non-completion, receives a larger penalty.  This is not a
    human quality label and the report says so explicitly.
    """
    items = row.get("input") or []
    names_by_id: dict[str | None, str] = {}
    results: list[tuple[str, bool]] = []
    for item in items:
        kind = item.get("type")
        call_id = item.get("call_id") or item.get("id")
        if kind in {"function_call", "custom_tool_call"}:
            names_by_id[call_id] = str(item.get("name", "unknown"))
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            results.append((names_by_id.get(call_id, "unknown"), tool_result_failed(item.get("output", ""))))

    last_by_tool: dict[str, bool] = {}
    for name, failed in results:
        last_by_tool[name] = failed
    failures = sum(failed for _, failed in results)
    unresolved = sum(last_by_tool.values())
    recovered = max(0, failures - unresolved)

    spans = list(spans if spans is not None else decompose(items))
    final_text = ""
    if spans:
        start, stop = spans[-1]
        final_text = " ".join(
            _content_text(item) for item in items[start:stop] if item.get("role") == "assistant"
        )
    negative_final = bool(NEGATIVE_FINAL_RE.search(final_text))

    result_denom = max(len(results), 1)
    tool_denom = max(len(last_by_tool), 1)
    score = (
        1.0
        - 0.35 * min(1.0, unresolved / tool_denom)
        - 0.08 * min(1.0, recovered / result_denom)
        - 0.25 * negative_final
    )
    return round(max(0.0, score), 6)


def build_trajectory(row: dict, row_idx: int, pricing=None) -> Trajectory:
    pricing = pricing or load_pricing()
    context = context_from_row(row, row_idx)
    items = row.get("input") or []
    spans = decompose(items)
    if not spans:
        raise ValueError(f"row {row_idx} has no recoverable model call")
    per_item = _item_tokens(items)
    prefix = [0]
    for tokens in per_item:
        prefix.append(prefix[-1] + tokens)
    tools_tokens = context.tools_tokens

    uncached = cached = output = 0
    previous_start = None
    for start, stop in spans:
        input_tokens = prefix[start] + tools_tokens
        cached_here = min(prefix[previous_start] + tools_tokens, input_tokens) if previous_start is not None else 0
        cached += cached_here
        uncached += input_tokens - cached_here
        output += prefix[stop] - prefix[start]
        previous_start = start

    pu, pc, po = price_of(str(row["model"]), pricing)
    cost = (uncached * pu + cached * pc + output * po) / 1e6
    return Trajectory(
        context=context,
        model=str(row["model"]),
        n_calls=len(spans),
        uncached_tokens=uncached,
        cached_tokens=cached,
        output_tokens=output,
        full_input_tokens=_json_tokens(items),
        observed_cost=cost,
        outcome=outcome_score(row, spans),
    )


def cost_for_model(record: Trajectory, model: str, pricing=None) -> float:
    """Hindsight same-trace cost.  Useful as a lower bound, not a causal estimate."""
    pu, pc, po = price_of(model, pricing or load_pricing())
    return (
        record.uncached_tokens * pu
        + record.cached_tokens * pc
        + record.output_tokens * po
    ) / 1e6


def load_rows(export_dir: str | Path) -> list[dict]:
    paths = sorted(Path(export_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl files under {export_dir}")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def load_trajectories(export_dir: str | Path, pricing=None) -> list[Trajectory]:
    pricing = pricing or load_pricing()
    return [build_trajectory(row, idx, pricing) for idx, row in enumerate(load_rows(export_dir))]


def _median_scale(values: Sequence[float]) -> tuple[float, float]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations) or 1.0
    return median, 1.4826 * mad


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


class MatchedResponseModel:
    """Inspectable response surface for token quantities, spend, and quality."""

    def __init__(self, records: Sequence[Trajectory], pricing=None, neighbors: int = 32):
        if not records:
            raise ValueError("cannot fit matcher without trajectories")
        self.records = list(records)
        self.pricing = pricing or load_pricing()
        self.neighbors = neighbors
        self.by_model: dict[str, list[Trajectory]] = {}
        for record in self.records:
            self.by_model.setdefault(record.model, []).append(record)

        dimensions = len(self.records[0].context.numeric)
        self.scales = [
            _median_scale([r.context.numeric[index] for r in self.records])
            for index in range(dimensions)
        ]
        document_frequency = Counter()
        for record in self.records:
            document_frequency.update(record.context.words)
        n = len(self.records)
        self.idf = {
            word: math.log((n + 1) / (count + 1)) + 1.0
            for word, count in document_frequency.items()
            if count >= 2 and count <= 0.75 * n
        }
        self.word_norm = {
            r.context.trajectory_id: math.sqrt(
                sum(self.idf[word] ** 2 for word in r.context.words if word in self.idf)
            ) or 1.0
            for r in self.records
        }
        self.priors = {model: self._prior(group) for model, group in self.by_model.items()}
        self.total_median_by_model = {
            model: statistics.median(record.total_tokens for record in group)
            for model, group in self.by_model.items()
        }
        self.total_median_by_family = {
            family: statistics.median(
                record.total_tokens for record in self.records if record.context.family == family
            )
            for family in SUPPORTED
            if any(record.context.family == family for record in self.records)
        }
        self.task_groups: dict[tuple[str, str], list[Trajectory]] = {}
        for record in self.records:
            key = (record.context.task_key, record.context.family)
            self.task_groups.setdefault(key, []).append(record)
        self.analogue_smear = self._fit_analogue_smear()
        self.analogue_total_smear = self._fit_analogue_total_smear()
        self.support_radius = {}
        for model, group in self.by_model.items():
            if len(group) < 3:
                self.support_radius[model] = float("inf")
                continue
            leave_one_out = []
            for record in group:
                leave_one_out.append(min(
                    self.distance(record.context, other.context)
                    for other in group if other is not record
                ))
            leave_one_out.sort()
            # A candidate is in support when it is no farther from a logged example
            # than 95% of that model's own logged contexts are from one another.
            self.support_radius[model] = leave_one_out[min(len(leave_one_out) - 1, math.ceil(0.95 * len(leave_one_out)) - 1)]

    @staticmethod
    def _prior(records: Sequence[Trajectory]) -> tuple[float, float, float, float]:
        return (
            statistics.mean(r.future_uncached_ratio for r in records),
            statistics.mean(r.cached_ratio for r in records),
            statistics.mean(r.output_ratio for r in records),
            statistics.mean(r.outcome for r in records),
        )

    @staticmethod
    def _targets(record: Trajectory) -> tuple[float, float, float, float]:
        return (
            record.future_uncached_ratio,
            record.cached_ratio,
            record.output_ratio,
            record.outcome,
        )

    def _raw_analogue(
        self, context: Context, model: str, exclude: Trajectory | None = None
    ) -> tuple[tuple[float, float, float, float], int] | None:
        pool = [
            record for record in self.task_groups.get((context.task_key, context.family), [])
            if record is not exclude
        ]
        if not pool or model not in self.priors:
            return None
        candidate_prior = self.priors[model]
        adjusted = [[], [], []]
        quality_residuals = []
        for record in pool:
            values = self._targets(record)
            source_prior = self.priors[record.model]
            for index in range(3):
                if source_prior[index] > 1e-9:
                    adjusted[index].append(values[index] * candidate_prior[index] / source_prior[index])
                else:
                    adjusted[index].append(values[index])
            quality_residuals.append(values[3] - source_prior[3])
        result = (
            statistics.median(adjusted[0]),
            statistics.median(adjusted[1]),
            statistics.median(adjusted[2]),
            max(0.0, min(1.0, candidate_prior[3] + statistics.median(quality_residuals))),
        )
        return result, len(pool)

    def _fit_analogue_smear(self) -> tuple[float, float, float]:
        """Training-only aggregate calibration for recurring-task analogue predictions."""
        actual_sums = [0.0, 0.0, 0.0]
        predicted_sums = [0.0, 0.0, 0.0]
        for group in self.task_groups.values():
            if len(group) < 2:
                continue
            for record in group:
                raw = self._raw_analogue(record.context, record.model, exclude=record)
                if not raw:
                    continue
                predicted, _ = raw
                actual = self._targets(record)
                for index in range(3):
                    actual_sums[index] += actual[index]
                    predicted_sums[index] += predicted[index]
        return tuple(
            max(0.75, min(1.50, actual_sums[index] / predicted_sums[index]))
            if predicted_sums[index] > 1e-9 else 1.0
            for index in range(3)
        )

    def _raw_analogue_total(
        self, context: Context, model: str, exclude: Trajectory | None = None
    ) -> tuple[float, int] | None:
        pool = [
            record for record in self.task_groups.get((context.task_key, context.family), [])
            if record is not exclude
        ]
        family_median = self.total_median_by_family.get(context.family)
        model_median = self.total_median_by_model.get(model)
        if not pool or not family_median or not model_median:
            return None
        adjusted = statistics.median(record.total_tokens for record in pool) * model_median / family_median
        return adjusted, len(pool)

    def _fit_analogue_total_smear(self) -> float:
        ratios = []
        for group in self.task_groups.values():
            if len(group) < 2:
                continue
            for record in group:
                raw = self._raw_analogue_total(record.context, record.model, exclude=record)
                if raw and raw[0] > 0:
                    ratios.append(record.total_tokens / raw[0])
        return max(0.75, min(1.50, statistics.median(ratios))) if ratios else 1.0

    def _cosine_distance(self, left: Context, right: Context) -> float:
        intersection = left.words & right.words
        dot = sum(self.idf.get(word, 0.0) ** 2 for word in intersection)
        left_norm = math.sqrt(sum(self.idf.get(word, 0.0) ** 2 for word in left.words)) or 1.0
        right_norm = self.word_norm.get(right.trajectory_id)
        if right_norm is None:
            right_norm = math.sqrt(sum(self.idf.get(word, 0.0) ** 2 for word in right.words)) or 1.0
        return 1.0 - max(0.0, min(1.0, dot / (left_norm * right_norm)))

    def distance(self, left: Context, right: Context) -> float:
        squared = []
        for index, (median, scale) in enumerate(self.scales):
            del median  # location cancels in a pairwise difference
            delta = min(abs(left.numeric[index] - right.numeric[index]) / scale, 4.0)
            squared.append((delta / 4.0) ** 2)
        numeric = math.sqrt(sum(squared) / len(squared))
        lexical = self._cosine_distance(left, right)
        tools = _jaccard_distance(left.tools, right.tools)
        tags = _jaccard_distance(left.tags, right.tags)
        return 0.50 * lexical + 0.25 * numeric + 0.15 * tools + 0.10 * tags

    def _nearest(self, context: Context, pool: Sequence[Trajectory], k: int) -> list[tuple[float, Trajectory]]:
        distances = [(self.distance(context, item.context), item) for item in pool]
        distances.sort(key=lambda pair: pair[0])
        return distances[: min(k, len(distances))]

    @staticmethod
    def _weights(neighbors: Sequence[tuple[float, Trajectory]]) -> list[float]:
        # A smooth local kernel avoids a single near-duplicate dominating the estimate.
        return [math.exp(-distance / 0.18) for distance, _ in neighbors]

    def predict(self, context: Context, model: str) -> Prediction:
        pool = self.by_model.get(model)
        if not pool:
            raise KeyError(f"model {model!r} has no support in the training fold")
        nearest = self._nearest(context, pool, self.neighbors)
        weights = self._weights(nearest)
        prior = self.priors[model]
        prior_strength = min(5.0, max(2.0, len(pool) / 40.0))
        weight_sum = sum(weights)

        def shrunk_mean(selector, prior_value):
            local = sum(weight * selector(record) for weight, (_, record) in zip(weights, nearest))
            return (local + prior_strength * prior_value) / (weight_sum + prior_strength)

        future_ratio = shrunk_mean(lambda r: r.future_uncached_ratio, prior[0])
        cached_ratio = shrunk_mean(lambda r: r.cached_ratio, prior[1])
        output_ratio = shrunk_mean(lambda r: r.output_ratio, prior[2])
        quality = shrunk_mean(lambda r: r.outcome, prior[3])
        # Conditional mean quantities above drive expected cost. Keep a separate
        # conditional median for a useful per-trajectory "typical length" forecast.
        general_p50_ratio = statistics.median(
            record.total_tokens / max(record.context.initial_tokens, 1)
            for _, record in nearest
        )
        total_tokens_p50 = context.initial_tokens * general_p50_ratio

        analogue = self._raw_analogue(context, model)
        analogue_count = 0
        if analogue:
            analogue_values, analogue_count = analogue
            # One prior run is informative; several recurring runs dominate the
            # generic matcher. The cap preserves some model-specific shrinkage.
            blend = min(0.85, analogue_count / (analogue_count + 1.0))
            future_ratio = (1.0 - blend) * future_ratio + blend * analogue_values[0] * self.analogue_smear[0]
            cached_ratio = (1.0 - blend) * cached_ratio + blend * analogue_values[1] * self.analogue_smear[1]
            output_ratio = (1.0 - blend) * output_ratio + blend * analogue_values[2] * self.analogue_smear[2]
            quality_blend = min(0.60, blend)
            quality = (1.0 - quality_blend) * quality + quality_blend * analogue_values[3]
        analogue_total = self._raw_analogue_total(context, model)
        if analogue_total:
            total_tokens_p50 = analogue_total[0] * self.analogue_total_smear

        effective = weight_sum * weight_sum / max(sum(w * w for w in weights), 1e-12)
        variance = sum(
            weight * (record.outcome - quality) ** 2
            for weight, (_, record) in zip(weights, nearest)
        ) / max(weight_sum, 1e-12)
        quality_se = math.sqrt((variance + 0.0025) / max(effective, 1.0))

        initial = context.initial_tokens
        uncached = initial * (1.0 + max(0.0, future_ratio))
        cached = initial * max(0.0, cached_ratio)
        output = initial * max(0.0, output_ratio)
        pu, pc, po = price_of(model, self.pricing)
        cost = (uncached * pu + cached * pc + output * po) / 1e6
        return Prediction(
            model=model,
            uncached_tokens=uncached,
            cached_tokens=cached,
            output_tokens=output,
            cost_usd=cost,
            quality=max(0.0, min(1.0, quality)),
            quality_se=quality_se,
            effective_neighbors=effective,
            nearest_distance=nearest[0][0],
            support_radius=self.support_radius[model],
            analogue_neighbors=analogue_count,
            total_tokens_p50=total_tokens_p50,
        )

    def propensities(self, context: Context, k: int = 80) -> dict[str, float]:
        pool = [record for record in self.records if record.context.family == context.family]
        nearest = self._nearest(context, pool, k)
        weights = self._weights(nearest)
        actions = sorted({record.model for record in pool})
        alpha = 0.75
        counts = {model: alpha for model in actions}
        for weight, (_, record) in zip(weights, nearest):
            counts[record.model] += weight
        total = sum(counts.values())
        return {model: count / total for model, count in counts.items()}


def _supported_predictions(matcher: MatchedResponseModel, context: Context) -> dict[str, Prediction]:
    predictions = {}
    for model in SUPPORTED[context.family]:
        if model in matcher.by_model:
            predictions[model] = matcher.predict(context, model)
    if not predictions:
        raise ValueError(f"no supported candidates for family {context.family}")
    return predictions


def choose_quality_constrained(
    predictions: dict[str, Prediction],
    tolerance: float,
    uncertainty_z: float = 0.0,
    min_effective_neighbors: float = 3.0,
    require_in_support: bool = False,
) -> str:
    """Cheapest candidate whose conservative quality remains near the best mean."""
    supported = [
        pred for pred in predictions.values()
        if pred.effective_neighbors >= min_effective_neighbors
        and (not require_in_support or pred.nearest_distance <= pred.support_radius)
    ] or list(predictions.values())
    best_mean = max(pred.quality for pred in supported)
    eligible = [
        pred for pred in supported
        if pred.quality - uncertainty_z * pred.quality_se >= best_mean - tolerance
    ]
    if not eligible:
        eligible = [max(supported, key=lambda pred: pred.quality - uncertainty_z * pred.quality_se)]
    return min(eligible, key=lambda pred: pred.cost_usd).model


class FittedRouter:
    """Deployable whole-trajectory router fitted on historical complete trajectories."""

    def __init__(self, matcher: MatchedResponseModel, version: str = "router_v4_conservative", workload_thresholds=None):
        self.matcher = matcher
        self.version = version
        self.workload_thresholds = workload_thresholds or {}

    def estimate(self, request: dict) -> dict[str, dict]:
        context = context_from_row(request)
        return {
            model: asdict(prediction)
            for model, prediction in _supported_predictions(self.matcher, context).items()
        }

    def route(self, request: dict) -> dict:
        context = context_from_row(request)
        predictions = _supported_predictions(self.matcher, context)
        action = choose_action(self.version, context, predictions, self.workload_thresholds)
        return {"model": action, "version": self.version, "estimates": {
            model: asdict(prediction) for model, prediction in predictions.items()
        }}


def fit_router(rows: Sequence[dict], version: str = "router_v4_conservative", pricing=None) -> FittedRouter:
    """Public training entry point.  `route()` never looks at training-row outcomes in its request."""
    pricing = pricing or load_pricing()
    records = [build_trajectory(row, idx, pricing) for idx, row in enumerate(rows)]
    matcher = MatchedResponseModel(records, pricing)
    thresholds = _workload_thresholds(records, matcher)
    return FittedRouter(matcher, version, thresholds)


def _workload_thresholds(records: Sequence[Trajectory], matcher: MatchedResponseModel | None = None) -> dict[str, float]:
    """Median workload on the same scale used by the deployable predictor.

    Comparing a model-specific prediction to the median factual quantity created a
    scale mismatch in v2's first iteration and upgraded every trajectory.  Fitting
    this threshold on training-only predictions makes the rule calibrated without
    looking at held-out futures.
    """
    result = {}
    for family in SUPPORTED:
        family_records = [r for r in records if r.context.family == family]
        if matcher and CHEAP[family] in matcher.by_model:
            predicted = [matcher.predict(r.context, CHEAP[family]) for r in family_records]
            totals = sorted(p.uncached_tokens + p.cached_tokens + p.output_tokens for p in predicted)
        else:
            totals = sorted(r.total_tokens for r in family_records)
        if totals:
            result[family] = float(totals[len(totals) // 2])
    return result


def choose_action(
    version: str,
    context: Context,
    predictions: dict[str, Prediction],
    workload_thresholds: dict[str, float],
) -> str:
    if version == "router_v2_workload":
        cheap = predictions.get(CHEAP[context.family]) or min(predictions.values(), key=lambda p: p.cost_usd)
        predicted_tokens = cheap.uncached_tokens + cheap.cached_tokens + cheap.output_tokens
        if predicted_tokens <= workload_thresholds.get(context.family, float("inf")):
            return cheap.model
        preferred = STRONG_DEFAULT[context.family]
        return preferred if preferred in predictions else max(predictions.values(), key=lambda p: p.quality).model
    if version == "router_v3_matched":
        return choose_quality_constrained(predictions, tolerance=0.008, uncertainty_z=0.0, min_effective_neighbors=3.0)
    if version == "router_v4_conservative":
        # Selected before any held-out chunk exists: retain a half-standard-error
        # guard while allowing a two-point proxy tolerance.  This is the knee of
        # the public-split frontier, not a claim of final external validation.
        return choose_quality_constrained(
            predictions, tolerance=0.020, uncertainty_z=0.5,
            min_effective_neighbors=5.0, require_in_support=True,
        )
    if version == "router_v5_near_oracle":
        return min(predictions.values(), key=lambda p: p.cost_usd).model
    if version == "router_v6_analogue":
        known_template = max(pred.analogue_neighbors for pred in predictions.values()) > 0
        return choose_quality_constrained(
            predictions,
            # Repeated jobs have directly observed workload analogues, so take a
            # more aggressive point on the frontier. Novel jobs exactly reuse v4.
            tolerance=0.040 if known_template else 0.020,
            uncertainty_z=0.25 if known_template else 0.50,
            min_effective_neighbors=5.0,
            require_in_support=True,
        )
    if version.startswith("frontier_"):
        tolerance = float(version.split("_", 1)[1])
        return choose_quality_constrained(
            predictions, tolerance=tolerance, uncertainty_z=0.5,
            min_effective_neighbors=5.0, require_in_support=True,
        )
    raise ValueError(f"unknown router version: {version}")


def _fold(context: Context, folds: int, group_tasks: bool = False) -> int:
    key = context.task_key if group_tasks else context.trajectory_id
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % folds


def crossfit(
    records: Sequence[Trajectory], versions: Sequence[str], pricing=None, folds: int = 5,
    group_tasks: bool = False,
) -> list[EvaluationRow]:
    """Produce out-of-fold response and propensity estimates for every trajectory."""
    pricing = pricing or load_pricing()
    output: list[EvaluationRow] = []
    for fold in range(folds):
        train = [record for record in records if _fold(record.context, folds, group_tasks) != fold]
        test = [record for record in records if _fold(record.context, folds, group_tasks) == fold]
        matcher = MatchedResponseModel(train, pricing)
        thresholds = _workload_thresholds(train, matcher) if "router_v2_workload" in versions else {}
        for record in test:
            predictions = _supported_predictions(matcher, record.context)
            actions = {
                version: choose_action(version, record.context, predictions, thresholds)
                for version in versions
            }
            # Re-application of the starter rule to one complete row.  It leaks final
            # size and depends on the behavior action, so it is never called DR below.
            actions["starter_baseline_corrected"] = (
                CHEAP[record.context.family]
                if record.full_input_tokens < 15_000
                else record.model
            )
            output.append(EvaluationRow(
                record=record,
                predictions=predictions,
                propensities=matcher.propensities(record.context),
                actions=actions,
            ))
    return sorted(output, key=lambda item: item.record.context.row)


def _bootstrap_ci(values: Sequence[float], seed: int, draws: int = 500) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def summarize_policy(evaluations: Sequence[EvaluationRow], version: str, propensity_floor: float = 0.05) -> dict:
    """Cross-fitted clipped DR value of a deterministic, action-independent policy."""
    dm_cost = []
    dm_quality = []
    dr_cost = []
    dr_quality = []
    cost_delta = []
    quality_delta = []
    trace_cost = []
    match_weights = []
    overlap = {}
    clipped = 0
    action_counts = Counter()
    pricing = load_pricing()
    for item in evaluations:
        action = item.actions[version]
        action_counts[action] += 1
        pred = item.predictions[action]
        dm_cost.append(pred.cost_usd)
        dm_quality.append(pred.quality)
        trace_cost.append(cost_for_model(item.record, action, pricing))
        influence_cost = pred.cost_usd
        influence_quality = pred.quality
        if item.record.model == action:
            raw_propensity = item.propensities.get(action, 0.0)
            propensity = max(raw_propensity, propensity_floor)
            clipped += raw_propensity < propensity_floor
            weight = 1.0 / propensity
            match_weights.append(weight)
            action_overlap = overlap.setdefault(action, {"selected": 0, "matches": 0, "weights": [], "propensities": []})
            action_overlap["matches"] += 1
            action_overlap["weights"].append(weight)
            action_overlap["propensities"].append(raw_propensity)
            influence_cost += weight * (item.record.observed_cost - pred.cost_usd)
            influence_quality += weight * (item.record.outcome - pred.quality)
        dr_cost.append(influence_cost)
        dr_quality.append(influence_quality)
        cost_delta.append(influence_cost - item.record.observed_cost)
        quality_delta.append(influence_quality - item.record.outcome)

        overlap.setdefault(action, {"selected": 0, "matches": 0, "weights": [], "propensities": []})["selected"] += 1

    n = len(evaluations)
    cost_low, cost_high = _bootstrap_ci(dr_cost, seed=20260822)
    quality_low, quality_high = _bootstrap_ci(dr_quality, seed=20260823)
    cost_delta_low, cost_delta_high = _bootstrap_ci(cost_delta, seed=20260824)
    quality_delta_low, quality_delta_high = _bootstrap_ci(quality_delta, seed=20260825)
    ess = (sum(match_weights) ** 2 / sum(w * w for w in match_weights)) if match_weights else 0.0
    action_overlap_summary = {}
    for action, values in sorted(overlap.items()):
        weights = values.pop("weights")
        propensities = values.pop("propensities")
        action_overlap_summary[action] = {
            **values,
            "match_ess": round((sum(weights) ** 2 / sum(w * w for w in weights)) if weights else 0.0, 2),
            "mean_match_propensity": round(statistics.mean(propensities), 4) if propensities else None,
        }
    return {
        "version": version,
        "estimator": "5-fold clipped doubly robust",
        "n_trajectories": n,
        "logged_matches": len(match_weights),
        "match_ess": round(ess, 2),
        "propensity_floor": propensity_floor,
        "propensities_clipped": clipped,
        "cost_dm_usd": round(sum(dm_cost), 4),
        "cost_dr_usd": round(statistics.mean(dr_cost) * n, 4),
        "cost_dr_ci_low_usd": round(cost_low * n, 4),
        "cost_dr_ci_high_usd": round(cost_high * n, 4),
        "cost_delta_vs_logged_usd": round(statistics.mean(cost_delta) * n, 4),
        "cost_delta_ci_low_usd": round(cost_delta_low * n, 4),
        "cost_delta_ci_high_usd": round(cost_delta_high * n, 4),
        "quality_dm": round(statistics.mean(dm_quality), 6),
        "quality_dr": round(statistics.mean(dr_quality), 6),
        "quality_dr_ci_low": round(quality_low, 6),
        "quality_dr_ci_high": round(quality_high, 6),
        "quality_delta_vs_logged": round(statistics.mean(quality_delta), 6),
        "quality_delta_ci_low": round(quality_delta_low, 6),
        "quality_delta_ci_high": round(quality_delta_high, 6),
        "same_trace_cost_usd": round(sum(trace_cost), 4),
        "action_counts": dict(sorted(action_counts.items())),
        "action_overlap": action_overlap_summary,
    }


def calibration_report(evaluations: Sequence[EvaluationRow]) -> dict:
    """Out-of-fold factual-action calibration; counterfactual calibration is unknowable."""
    rows = []
    by_model = {}
    for item in evaluations:
        prediction = item.predictions.get(item.record.model)
        if not prediction:
            continue
        actual_cost = item.record.observed_cost
        predicted_tokens = prediction.uncached_tokens + prediction.cached_tokens + prediction.output_tokens
        actual_tokens = item.record.total_tokens
        values = {
            "model": item.record.model,
            "known_template": prediction.analogue_neighbors > 0,
            "actual_cost": actual_cost,
            "predicted_cost": prediction.cost_usd,
            "cost_ape": abs(prediction.cost_usd - actual_cost) / max(actual_cost, 1e-12),
            "cost_ratio": prediction.cost_usd / max(actual_cost, 1e-12),
            "actual_tokens": actual_tokens,
            "predicted_tokens": predicted_tokens,
            "tokens_ape": abs(predicted_tokens - actual_tokens) / max(actual_tokens, 1),
            "predicted_tokens_p50": prediction.total_tokens_p50,
            "tokens_p50_ape": abs(prediction.total_tokens_p50 - actual_tokens) / max(actual_tokens, 1),
            "quality_error": prediction.quality - item.record.outcome,
        }
        rows.append(values)
        by_model.setdefault(item.record.model, []).append(values)

    def summarize(values):
        actual_cost_total = sum(v["actual_cost"] for v in values)
        actual_tokens_total = sum(v["actual_tokens"] for v in values)
        return {
            "n": len(values),
            "cost_median_absolute_pct_error": round(100 * statistics.median(v["cost_ape"] for v in values), 2),
            "cost_mean_predicted_to_actual_ratio": round(statistics.mean(v["cost_ratio"] for v in values), 4),
            "cost_aggregate_predicted_to_actual_ratio": round(
                sum(v["predicted_cost"] for v in values) / max(actual_cost_total, 1e-12), 4
            ),
            "cost_weighted_absolute_pct_error": round(
                100 * sum(abs(v["predicted_cost"] - v["actual_cost"]) for v in values) / max(actual_cost_total, 1e-12), 2
            ),
            "tokens_median_absolute_pct_error": round(100 * statistics.median(v["tokens_ape"] for v in values), 2),
            "tokens_p50_median_absolute_pct_error": round(
                100 * statistics.median(v["tokens_p50_ape"] for v in values), 2
            ),
            "tokens_aggregate_predicted_to_actual_ratio": round(
                sum(v["predicted_tokens"] for v in values) / max(actual_tokens_total, 1), 4
            ),
            "tokens_weighted_absolute_pct_error": round(
                100 * sum(abs(v["predicted_tokens"] - v["actual_tokens"]) for v in values) / max(actual_tokens_total, 1), 2
            ),
            "quality_mean_error": round(statistics.mean(v["quality_error"] for v in values), 6),
            "quality_rmse": round(math.sqrt(statistics.mean(v["quality_error"] ** 2 for v in values)), 6),
        }

    return {
        "scope": "out-of-fold predictions evaluated only where the factual model is a supported candidate",
        "warning": "factual calibration does not validate counterfactual predictions",
        "overall": summarize(rows),
        "by_model": {model: summarize(values) for model, values in sorted(by_model.items())},
        "by_template_status": {
            "known": summarize([value for value in rows if value["known_template"]]),
            "novel": summarize([value for value in rows if not value["known_template"]]),
        },
    }


def summarize_starter(evaluations: Sequence[EvaluationRow]) -> dict:
    """Model-assisted switch estimate; deliberately not mislabelled as DR."""
    costs = []
    qualities = []
    trace_cost = []
    action_counts = Counter()
    factual = 0
    pricing = load_pricing()
    for item in evaluations:
        action = item.actions["starter_baseline_corrected"]
        action_counts[action] += 1
        if action == item.record.model:
            costs.append(item.record.observed_cost)
            qualities.append(item.record.outcome)
            factual += 1
        else:
            prediction = item.predictions[action]
            costs.append(prediction.cost_usd)
            qualities.append(prediction.quality)
        trace_cost.append(cost_for_model(item.record, action, pricing))
    return {
        "version": "starter_baseline_corrected",
        "estimator": "cross-fitted model-assisted switch (not DR; policy uses logged action/final size)",
        "n_trajectories": len(evaluations),
        "factual_trajectories": factual,
        "cost_est_usd": round(sum(costs), 4),
        "quality_est": round(statistics.mean(qualities), 6),
        "same_trace_cost_usd": round(sum(trace_cost), 4),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _write_routes(results_root: Path, evaluations: Sequence[EvaluationRow], version: str, summary: dict) -> None:
    directory = results_root / version
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "routes.jsonl").open("w") as handle:
        for item in evaluations:
            action = item.actions[version]
            prediction = item.predictions.get(action)
            estimated_cost = prediction.cost_usd if prediction else item.record.observed_cost
            handle.write(json.dumps({
                "trajectory": item.record.context.trajectory_id,
                "row": item.record.context.row,
                "n_calls": item.record.n_calls,
                "logged_model": item.record.model,
                "route": [action] * item.record.n_calls,
                "routed_model": action,
                "cost_logged_usd": round(item.record.observed_cost, 8),
                "cost_routed_usd": round(estimated_cost, 8),
                "cost_routed_same_trace_usd": round(cost_for_model(item.record, action), 8),
                "quality_logged_proxy": item.record.outcome,
                "quality_routed_est": round(prediction.quality, 6) if prediction else item.record.outcome,
                "uncached_tokens_routed_est": round(prediction.uncached_tokens) if prediction else None,
                "cached_tokens_routed_est": round(prediction.cached_tokens) if prediction else None,
                "output_tokens_routed_est": round(prediction.output_tokens) if prediction else None,
                "total_tokens_p50_est": round(prediction.total_tokens_p50) if prediction else None,
                "analogue_neighbors": prediction.analogue_neighbors if prediction else 0,
                "effective_neighbors": round(prediction.effective_neighbors, 2) if prediction else None,
                "switches": 0,
                "estimate_is_cross_fitted": True,
            }) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _existing_starter_totals(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return {
        "note": "starter's original 953-group, input-only accounting; not comparable to corrected ledger",
        "groups": len(rows),
        "logged_input_only_usd": round(sum(row["cost_logged_usd"] for row in rows), 4),
        "routed_input_only_usd": round(sum(row["cost_routed_usd"] for row in rows), 4),
    }


def _write_comparison(destination: Path, summaries: Sequence[dict], logged_cost: float, logged_quality: float) -> None:
    fields = [
        "version", "estimator", "cost_usd", "cost_ci_low_usd", "cost_ci_high_usd",
        "saving_vs_logged_pct", "cost_delta_ci_low_usd", "cost_delta_ci_high_usd",
        "quality", "quality_ci_low", "quality_ci_high", "quality_delta_vs_logged",
        "quality_delta_ci_low", "quality_delta_ci_high",
        "logged_matches", "match_ess", "same_trace_cost_usd", "action_counts",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "version": "logged", "estimator": "factual", "cost_usd": round(logged_cost, 4),
            "saving_vs_logged_pct": 0.0, "quality": round(logged_quality, 6),
        })
        for summary in summaries:
            cost = summary.get("cost_dr_usd", summary.get("cost_est_usd"))
            quality = summary.get("quality_dr", summary.get("quality_est"))
            writer.writerow({
                "version": summary["version"],
                "estimator": summary["estimator"],
                "cost_usd": cost,
                "cost_ci_low_usd": summary.get("cost_dr_ci_low_usd", ""),
                "cost_ci_high_usd": summary.get("cost_dr_ci_high_usd", ""),
                "saving_vs_logged_pct": round((cost / logged_cost - 1.0) * 100.0, 2),
                "cost_delta_ci_low_usd": summary.get("cost_delta_ci_low_usd", ""),
                "cost_delta_ci_high_usd": summary.get("cost_delta_ci_high_usd", ""),
                "quality": quality,
                "quality_ci_low": summary.get("quality_dr_ci_low", ""),
                "quality_ci_high": summary.get("quality_dr_ci_high", ""),
                "quality_delta_vs_logged": summary.get("quality_delta_vs_logged", ""),
                "quality_delta_ci_low": summary.get("quality_delta_ci_low", ""),
                "quality_delta_ci_high": summary.get("quality_delta_ci_high", ""),
                "logged_matches": summary.get("logged_matches", summary.get("factual_trajectories", "")),
                "match_ess": summary.get("match_ess", ""),
                "same_trace_cost_usd": summary.get("same_trace_cost_usd", ""),
                "action_counts": json.dumps(summary.get("action_counts", {}), sort_keys=True),
            })


def _write_report(
    path: Path,
    records: Sequence[Trajectory],
    summaries: Sequence[dict],
    starter_original: dict | None,
    hindsight_oracle: float,
    calibration: dict,
    grouped_summaries: Sequence[dict],
    sensitivity_summaries: Sequence[dict],
) -> None:
    logged_cost = sum(record.observed_cost for record in records)
    logged_quality = statistics.mean(record.outcome for record in records)
    recommended = next(item for item in summaries if item["version"] == "router_v4_conservative")
    analogue = next(item for item in summaries if item["version"] == "router_v6_analogue")
    grouped_analogue = next(item for item in grouped_summaries if item["version"] == "router_v6_analogue")
    sensitivity_costs = [item["cost_dr_usd"] for item in sensitivity_summaries]
    lines = [
        "# Oracle-router evaluation",
        "",
        f"Corrected ledger: {len(records):,} complete trajectories, "
        f"{sum(r.n_calls for r in records):,} reconstructed calls.",
        f"Logged factual spend: **${logged_cost:,.2f}**; execution proxy: **{logged_quality:.4f}**.",
        f"Recommended v4 clipped-DR spend: **${recommended['cost_dr_usd']:,.2f}** "
        f"[{recommended['cost_dr_ci_low_usd']:,.2f}, {recommended['cost_dr_ci_high_usd']:,.2f}], "
        f"proxy **{recommended['quality_dr']:.4f}** "
        f"[{recommended['quality_dr_ci_low']:.4f}, {recommended['quality_dr_ci_high']:.4f}].",
        f"Paired v4 quality difference vs logged: **{recommended['quality_delta_vs_logged']:+.4f}** "
        f"[{recommended['quality_delta_ci_low']:+.4f}, {recommended['quality_delta_ci_high']:+.4f}].",
        f"Same-trace, same-family hindsight cost floor: **${hindsight_oracle:,.2f}** "
        "(not a quality-aware causal oracle).",
        f"Factual-action token prediction median absolute percentage error: "
        f"**{calibration['overall']['tokens_median_absolute_pct_error']:.1f}%**; "
        f"cost error: **{calibration['overall']['cost_median_absolute_pct_error']:.1f}%**.",
        f"Separate typical-length (p50) token MdAPE: "
        f"**{calibration['overall']['tokens_p50_median_absolute_pct_error']:.1f}%**.",
        f"Aggregate factual-action predicted/actual ratio: "
        f"**{calibration['overall']['cost_aggregate_predicted_to_actual_ratio']:.3f}x cost**, "
        f"**{calibration['overall']['tokens_aggregate_predicted_to_actual_ratio']:.3f}x tokens**.",
        f"Analogue-aware v6 spend: **${analogue['cost_dr_usd']:,.2f}**; on task-grouped "
        f"validation (no normalized template crosses folds): **${grouped_analogue['cost_dr_usd']:,.2f}**.",
        f"v4 propensity-floor sensitivity (0.02 to 0.20): "
        f"**${min(sensitivity_costs):,.2f}–${max(sensitivity_costs):,.2f}**.",
        "",
        "## What is honest here",
        "",
        "- Routing features stop before the first model-produced item; future length never enters v2-v4.",
        "- Every policy estimate is out-of-fold. Propensities are local matched frequencies and are clipped at 0.05.",
        "- A second five-fold protocol groups normalized task templates, stress-testing unseen-task generalization.",
        "- Routing stays within the existing Claude/GPT harness family; cross-family overlap is effectively absent.",
        "- Spend includes inferred cache reads and recovered output tokens at official rates.",
        "- Tokens are chars/4 estimates, not provider usage. The quality target is an execution-completion proxy, not human judgment.",
        "",
        "## Main failure modes",
        "",
        "- Logged model choice is observational and may remain confounded after matching.",
        "- Counterfactual models can change the number of calls, tool results, and context; the response surface only learns this where there is overlap.",
        "- GPT final assistant text is often blank in this export, so the proxy intentionally relies mostly on tool completion.",
        "- Luna has only 20 logged trajectories; uncertainty and clipped-weight intervals should be taken seriously.",
        "- Calibration can only be checked for the model that actually ran; it does not prove counterfactual calibration.",
        "- The same public chunk was used to select the policy class. A future chunk must be treated as untouched validation.",
    ]
    if starter_original:
        lines += [
            "",
            "## Original starter artifact (not apples-to-apples)",
            "",
            f"It reports {starter_original['groups']} reconstructed groups, "
            f"${starter_original['logged_input_only_usd']:.2f} logged input-only and "
            f"${starter_original['routed_input_only_usd']:.2f} routed input-only. "
            "The real row-level ledger has 1,000 trajectories and includes recovered output cost.",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    pricing = load_pricing()
    records = load_trajectories(export, pricing)
    versions = [
        "router_v2_workload",
        "router_v3_matched",
        "router_v4_conservative",
        "router_v5_near_oracle",
        "router_v6_analogue",
    ]
    frontier_versions = [f"frontier_{value}" for value in ("0.0", "0.005", "0.01", "0.02", "0.04", "1.0")]
    evaluations = crossfit(records, versions + frontier_versions, pricing)
    calibration = calibration_report(evaluations)

    grouped_versions = ["router_v4_conservative", "router_v5_near_oracle", "router_v6_analogue"]
    grouped_evaluations = crossfit(records, grouped_versions, pricing, group_tasks=True)
    grouped_summaries = [summarize_policy(grouped_evaluations, version) for version in grouped_versions]
    for summary in grouped_summaries:
        summary["estimator"] = "5-fold task-grouped clipped doubly robust"

    summaries = [summarize_policy(evaluations, version) for version in versions]
    starter = summarize_starter(evaluations)
    comparison_summaries = [starter] + summaries
    sensitivity_summaries = []
    for floor in (0.02, 0.05, 0.10, 0.20):
        summary = summarize_policy(evaluations, "router_v4_conservative", propensity_floor=floor)
        summary["version"] = f"router_v4_propensity_floor_{floor:.2f}"
        sensitivity_summaries.append(summary)

    repo = SCRIPTS.parent
    results_root = repo / "results"
    results_root.mkdir(exist_ok=True)
    for summary in comparison_summaries:
        _write_routes(results_root, evaluations, summary["version"], summary)

    artifact = results_root / "oracle_router"
    artifact.mkdir(exist_ok=True)
    logged_cost = sum(record.observed_cost for record in records)
    logged_quality = statistics.mean(record.outcome for record in records)
    _write_comparison(artifact / "comparison.csv", comparison_summaries, logged_cost, logged_quality)
    _write_comparison(artifact / "grouped_validation.csv", grouped_summaries, logged_cost, logged_quality)
    _write_comparison(artifact / "sensitivity.csv", sensitivity_summaries, logged_cost, logged_quality)

    frontier = [summarize_policy(evaluations, version) for version in frontier_versions]
    _write_comparison(artifact / "frontier.csv", frontier, logged_cost, logged_quality)

    hindsight_oracle = sum(
        min(cost_for_model(record, model, pricing) for model in SUPPORTED[record.context.family])
        for record in records
    )
    original = _existing_starter_totals(results_root / "routes.jsonl")
    metadata = {
        "token_accounting": "serialized chars / 4 estimates",
        "cost_scope": "cache-aware input + recovered output",
        "logged_factual_cost_usd": round(logged_cost, 4),
        "logged_quality_proxy": round(logged_quality, 6),
        "same_trace_same_family_hindsight_floor_usd": round(hindsight_oracle, 4),
        "original_starter_artifact": original,
        "calibration": calibration,
        "summaries": comparison_summaries,
        "task_grouped_summaries": grouped_summaries,
        "propensity_sensitivity": sensitivity_summaries,
    }
    (artifact / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (artifact / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    _write_report(
        artifact / "REPORT.md", records, comparison_summaries, original,
        hindsight_oracle, calibration, grouped_summaries, sensitivity_summaries,
    )

    print(f"rows={len(records):,} reconstructed calls={sum(r.n_calls for r in records):,}")
    print(f"logged factual cost (estimated tokens, cache-aware input+output): ${logged_cost:,.2f}")
    for summary in comparison_summaries:
        cost = summary.get("cost_dr_usd", summary.get("cost_est_usd"))
        quality = summary.get("quality_dr", summary.get("quality_est"))
        print(f"{summary['version']:28s} cost=${cost:8.2f}  quality_proxy={quality:.4f}")
    for summary in grouped_summaries:
        print(f"grouped {summary['version']:20s} cost=${summary['cost_dr_usd']:8.2f}  "
              f"quality_proxy={summary['quality_dr']:.4f}")
    print(f"same-trace same-family hindsight floor: ${hindsight_oracle:,.2f}")
    print("factual-action calibration: "
          f"token MdAPE={calibration['overall']['tokens_median_absolute_pct_error']:.1f}%  "
          f"cost MdAPE={calibration['overall']['cost_median_absolute_pct_error']:.1f}%")
    print(f"wrote {artifact / 'comparison.csv'}, {artifact / 'grouped_validation.csv'}, "
          f"{artifact / 'sensitivity.csv'}, {artifact / 'frontier.csv'}, and {artifact / 'REPORT.md'}")
    print("NOTE: all tokens are chars/4 estimates; quality is a proxy; final-call counterfactuals remain unobserved.")


if __name__ == "__main__":
    main()
