"""Aggregations behind the explorer UI.

Every number here is derived from two local files and nothing else:

  export/*.jsonl    the trajectories — one row is one COMPLETE trajectory, unfolded
                    by scripts/decompose.py into the calls that actually ran
  results/*.jsonl   a router's decision per call, joined on trajectory id

All token counts are ESTIMATES (serialized JSON chars / 4): the export ships no
`usage` field. Outputs are recovered from the next call's input, so every call has
an observed output except the last one of each trajectory.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Which axis a trajectory can be classified on. `column` is the trajectory_meta or
# trajectories column; `limit` caps high-cardinality axes to their top groups.
CLASS_AXES: List[Dict[str, Any]] = [
    {"key": "model", "label": "Logged model", "source": "t",
     "help": "The model that actually served every call of the trajectory. The premise "
             "is one model per trajectory; the variation ACROSS trajectories is the "
             "natural experiment this whole evaluation rests on."},
    {"key": "workload", "label": "Workload", "source": "m",
     "help": "The kind of work the model did most of, from the tools it actually "
             "called: exec (bash/shell), edit (patch/write), read, message, wait."},
    {"key": "trigger_kind", "label": "Trigger", "source": "m",
     "help": "Who started the run, read out of the `Triggered by:` line in the opening "
             "message: a cron, a trigger, or a human user."},
    {"key": "harness", "label": "Harness", "source": "m",
     "help": "Which tool dialect the agent was given: bash+file_edit, or "
             "shell_command+apply_patch. A structural split, not a guess."},
    {"key": "surface", "label": "Chat surface", "source": "m",
     "help": "Where the agent reports back — Slack or MS Teams — from the coworker_* "
             "tools in the request's tool list."},
    {"key": "size_band", "label": "Size band", "source": "m",
     "help": "Total estimated billed input tokens for the trajectory: "
             "XS <10k, S <50k, M <200k, L <1M, XL >=1M."},
    {"key": "depth_band", "label": "Depth band", "source": "m",
     "help": "How many calls the agent loop ran: single, short (<=4), medium (<=9), "
             "long (<=24), deep (25+)."},
    {"key": "toolset_label", "label": "Toolset", "source": "m",
     "help": "The exact set of tool definitions sent with the request. Only 8 distinct "
             "sets exist across the corpus — a clean structural class."},
    {"key": "cron_family", "label": "Cron family", "source": "m", "limit": 25,
     "help": "Leading segment of the cron path. CAVEAT: redaction placeholders are only "
             "consistent WITHIN a trajectory, so families that are themselves a PII_* "
             "token merge unrelated crons. Those rows are flagged."},
    {"key": "shape_signature", "label": "Loop shape", "source": "m", "limit": 25,
     "help": "Run-length-compressed sequence of turn kinds: T tool call, M multi-tool, "
             "P patch, S text, X the final unobservable turn. Same signature means the "
             "two runs took structurally the same path through the agent loop."},
]
AXIS_BY_KEY = {axis["key"]: axis for axis in CLASS_AXES}

# scripts/decompose.py groups tool names the same way.
TOOL_WORK = {
    "bash": "exec", "shell_command": "exec", "run_shell": "exec",
    "apply_patch": "edit", "file_edit": "edit", "file_write": "edit",
    "file_read": "read", "view_image": "read", "memory_search": "read",
    "coworker_channel_history": "read", "coworker_download_from_slack": "read",
    "coworker_download_from_channel": "read", "coworker_get_channel_reactions": "read",
    "coworker_list_msteams_channels": "read", "coworker_list_msteams_users": "read",
    "coworker_send_slack_message": "message", "coworker_send_msteams_message": "message",
    "send_message_to_thread": "message", "coworker_slack_react": "message",
    "coworker_msteams_react": "message", "coworker_upload_to_slack": "message",
    "coworker_upload_to_channel": "message", "coworker_delete_channel_message": "message",
    "coworker_display_ui_elements": "message", "submit_draft": "message",
    "submit_subagent_result": "message", "wait_for_background_work": "wait",
}


def rows(cursor: Any) -> List[Dict[str, Any]]:
    names = [description[0] for description in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def overview(con: Any) -> Dict[str, Any]:
    headline = rows(con.execute("""
        SELECT count(*) AS calls,
               count(DISTINCT trajectory_id) AS trajectories,
               count(DISTINCT model) AS models,
               sum(est_input_tokens + coalesce(tools_tokens, 0)) AS billed_input_tokens,
               sum(shared_prefix_tokens) AS cached_tokens,
               sum(est_output_tokens) AS output_tokens,
               sum(logged_input_cost_usd) AS input_cost_usd,
               sum(logged_output_cost_usd) AS output_cost_usd,
               sum(logged_total_cost_usd) AS logged_cost_usd,
               sum(cost_usd) AS routed_cost_usd,
               count(chosen_model) AS routed_calls,
               count(*) FILTER (WHERE est_output_tokens IS NULL) AS unobservable_calls
        FROM calls"""))[0]
    billed = headline["billed_input_tokens"] or 0
    headline["cache_hit_rate"] = (headline["cached_tokens"] or 0) / billed if billed else None
    logged, routed = headline["logged_cost_usd"], headline["routed_cost_usd"]
    headline["routed_delta_pct"] = (routed / logged - 1) if logged and routed else None

    by_model = rows(con.execute("""
        SELECT c.model, c.model_family, p.tier_rank, p.uncached, p.cached, p.output,
               count(*) AS calls, count(DISTINCT c.trajectory_id) AS trajectories,
               sum(c.logged_total_cost_usd) AS cost_usd,
               sum(c.est_input_tokens + coalesce(c.tools_tokens, 0)) AS billed_input_tokens,
               sum(c.est_output_tokens) AS output_tokens,
               sum(c.shared_prefix_tokens) AS cached_tokens
        FROM calls c LEFT JOIN model_prices p ON p.model = c.model
        GROUP BY ALL ORDER BY cost_usd DESC"""))

    # Cold start is the biggest single line item: call 0 has no cache to hit.
    by_call_index = rows(con.execute("""
        SELECT least(call_index, 20) AS call_index,
               count(*) AS calls, sum(logged_total_cost_usd) AS cost_usd,
               avg(prefix_share) AS mean_prefix_share
        FROM calls GROUP BY ALL ORDER BY call_index"""))

    concentration = rows(con.execute("""
        WITH ranked AS (
          SELECT logged_total_cost_usd AS cost_usd,
                 ntile(10) OVER (ORDER BY logged_total_cost_usd DESC) AS decile
          FROM trajectories WHERE logged_total_cost_usd IS NOT NULL)
        SELECT decile, count(*) AS trajectories, sum(cost_usd) AS cost_usd
        FROM ranked GROUP BY decile ORDER BY decile"""))

    by_output_kind = rows(con.execute("""
        SELECT coalesce(output_kind, 'unobservable') AS output_kind, count(*) AS calls,
               sum(logged_total_cost_usd) AS cost_usd,
               avg(est_output_tokens) AS mean_output_tokens
        FROM calls GROUP BY ALL ORDER BY cost_usd DESC"""))

    sources = rows(con.execute("""
        SELECT kind, path, line_count, parse_errors, size_bytes FROM sources
        ORDER BY kind, path"""))
    return {
        "headline": headline, "by_model": by_model, "by_call_index": by_call_index,
        "concentration": concentration, "by_output_kind": by_output_kind,
        "sources": sources, "route_join": route_join(con),
    }


def route_join(con: Any) -> Dict[str, Any]:
    """Did results/*.jsonl actually attach to the calls, and if not, why not.

    A router artifact carries its own trajectory ids. Those only mean anything if
    they were minted from the same reconstruction the calls came from. When they
    were not, the join silently yields nothing and every route column reads null —
    which looks identical to "no router has run yet". Report the difference.
    """
    ledger = rows(con.execute("""
        SELECT count(*) AS entries,
               count(DISTINCT trajectory_id) AS trajectory_ids
        FROM ledger_entries"""))[0]
    matched = rows(con.execute("""
        SELECT count(DISTINCT l.trajectory_id) AS matched_trajectories
        FROM ledger_entries l JOIN trajectories t USING(trajectory_id)"""))[0]
    joined_calls = con.execute(
        "SELECT count(chosen_model) FROM calls").fetchone()[0]
    unmatched = [row[0] for row in con.execute("""
        SELECT DISTINCT l.trajectory_id FROM ledger_entries l
        LEFT JOIN trajectories t USING(trajectory_id)
        WHERE t.trajectory_id IS NULL LIMIT 4""").fetchall()]
    expected = [row[0] for row in con.execute(
        "SELECT trajectory_id FROM trajectories LIMIT 4").fetchall()]
    return {
        "result_entries": ledger["entries"],
        "result_trajectory_ids": ledger["trajectory_ids"],
        "matched_trajectories": matched["matched_trajectories"],
        "joined_calls": joined_calls,
        "sample_unmatched_ids": unmatched,
        "sample_expected_ids": expected,
        "healthy": bool(ledger["entries"]) and matched["matched_trajectories"] > 0,
        "present": bool(ledger["entries"]),
    }


def _axis_sql(axis: Dict[str, Any]) -> str:
    return "%s.%s" % (axis["source"], axis["key"])


def classes(con: Any, axis_key: Optional[str] = None) -> Dict[str, Any]:
    """One frequency/cost table per classification axis."""
    wanted = [AXIS_BY_KEY[axis_key]] if axis_key else CLASS_AXES
    out = []
    for axis in wanted:
        limit = axis.get("limit")
        table = rows(con.execute("""
            SELECT coalesce(CAST(%s AS VARCHAR), '(none)') AS value,
                   count(*) AS trajectories, sum(t.n_calls) AS calls,
                   sum(t.logged_total_cost_usd) AS cost_usd,
                   sum(t.routed_cost_usd) AS routed_cost_usd,
                   avg(t.logged_total_cost_usd) AS mean_cost_usd,
                   median(t.logged_total_cost_usd) AS median_cost_usd,
                   median(t.n_calls) AS median_calls,
                   avg(m.cache_hit_rate) AS cache_hit_rate,
                   count(*) FILTER (WHERE t.has_error) AS error_trajectories,
                   count(DISTINCT t.model) AS distinct_models
            FROM trajectories t JOIN trajectory_meta m USING(trajectory_id)
            GROUP BY 1 ORDER BY cost_usd DESC %s""" % (
                _axis_sql(axis), ("LIMIT %d" % int(limit)) if limit else "")))
        total = sum(row["cost_usd"] or 0 for row in table)
        for row in table:
            row["cost_share"] = (row["cost_usd"] or 0) / total if total else None
            logged, routed = row["cost_usd"], row["routed_cost_usd"]
            row["routed_delta_pct"] = (routed / logged - 1) if logged and routed else None
        out.append({**{k: v for k, v in axis.items() if k != "source"}, "rows": table})
    return {"axes": out}


def tool_profile(con: Any) -> Dict[str, Any]:
    """Function calls, classified — and what each one costs downstream.

    A tool's real cost is not the call that requests it; it is the tokens its RESULT
    adds to every later prompt in the trajectory. `est_new_tokens` on the call after
    a tool result is exactly that growth, so grouping it by `last_tool_name`
    attributes prompt growth to the tool that caused it.
    """
    tools = rows(con.execute("""
        SELECT last_tool_name AS tool, count(*) AS calls_after,
               count(DISTINCT trajectory_id) AS trajectories,
               sum(est_new_tokens) AS added_prompt_tokens,
               avg(est_new_tokens) AS mean_added_tokens,
               median(est_new_tokens) AS median_added_tokens,
               max(est_new_tokens) AS max_added_tokens,
               sum(logged_total_cost_usd) AS cost_usd,
               avg(CAST(tool_output_error AS DOUBLE)) AS error_rate,
               avg(CAST(repeat_tool AS DOUBLE)) AS repeat_rate
        FROM calls WHERE last_tool_name IS NOT NULL
        GROUP BY 1 ORDER BY added_prompt_tokens DESC"""))
    issued = dict(con.execute("""
        SELECT tool, count(*) FROM (SELECT unnest(output_tool_names) AS tool FROM calls)
        WHERE tool IS NOT NULL GROUP BY 1""").fetchall())
    total_added = sum(row["added_prompt_tokens"] or 0 for row in tools)
    for row in tools:
        row["work"] = TOOL_WORK.get(row["tool"], "other")
        row["times_issued"] = issued.get(row["tool"])
        row["added_share"] = (row["added_prompt_tokens"] or 0) / total_added if total_added else None
    by_work: Dict[str, Dict[str, Any]] = {}
    for row in tools:
        bucket = by_work.setdefault(row["work"], {
            "work": row["work"], "tools": 0, "calls_after": 0,
            "added_prompt_tokens": 0, "cost_usd": 0.0,
        })
        bucket["tools"] += 1
        bucket["calls_after"] += row["calls_after"] or 0
        bucket["added_prompt_tokens"] += row["added_prompt_tokens"] or 0
        bucket["cost_usd"] += row["cost_usd"] or 0.0
    return {
        "tools": tools,
        "by_work": sorted(by_work.values(), key=lambda b: -b["added_prompt_tokens"]),
        "total_added_prompt_tokens": total_added,
    }


FILTERABLE = {
    "model": "t.model", "model_family": "t.model_family",
    "workload": "m.workload", "trigger_kind": "m.trigger_kind",
    "harness": "m.harness", "surface": "m.surface", "size_band": "m.size_band",
    "depth_band": "m.depth_band", "toolset_label": "m.toolset_label",
    "cron_family": "m.cron_family", "shape_signature": "m.shape_signature",
}
SORTABLE = {
    "cost_usd": "t.logged_total_cost_usd", "n_calls": "t.n_calls",
    "billed_input_tokens": "m.billed_input_tokens",
    "output_tokens": "t.total_est_output_tokens",
    "cache_hit_rate": "m.cache_hit_rate",
    "saving_usd": "(t.logged_total_cost_usd - t.routed_cost_usd)",
    "started_at": "m.started_at", "trajectory_id": "t.trajectory_id",
}
TRAJECTORY_SELECT = """
    SELECT t.trajectory_id, t.model, t.model_family, t.n_calls,
           t.total_est_input_tokens, t.total_est_output_tokens,
           t.total_shared_prefix_tokens, t.tools_tokens,
           t.logged_cost_usd AS input_cost_usd,
           t.logged_output_cost_usd AS output_cost_usd,
           t.logged_total_cost_usd AS cost_usd, t.routed_cost_usd,
           t.cold_start_cost_usd, t.has_error, t.has_repeat_tool, t.n_error_calls,
           t.source_path, t.source_line,
           m.workload, m.trigger_kind, m.harness, m.surface, m.toolset_label,
           m.cron_family, m.cron_family_collapsed, m.cron_path, m.thread_path,
           m.started_at, m.size_band, m.depth_band, m.shape_signature,
           m.tool_signature, m.n_tool_calls, m.work_mix, m.billed_input_tokens,
           m.cache_hit_rate
    FROM trajectories t JOIN trajectory_meta m USING(trajectory_id)"""


def _where(filters: Dict[str, str], search: Optional[str]) -> Tuple[str, List[Any]]:
    clauses, params = [], []
    for key, value in filters.items():
        if value is None or key not in FILTERABLE:
            continue
        clauses.append("%s = ?" % FILTERABLE[key])
        params.append(value)
    if search:
        clauses.append("(t.trajectory_id ILIKE ? OR t.opening_user ILIKE ? "
                       "OR m.cron_path ILIKE ? OR m.tool_signature ILIKE ?)")
        params.extend(["%%%s%%" % search] * 4)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def trajectory_list(
    con: Any, filters: Dict[str, str], search: Optional[str] = None,
    sort: str = "cost_usd", order: str = "desc", limit: int = 100, offset: int = 0,
) -> Dict[str, Any]:
    where, params = _where(filters, search)
    sort_sql = SORTABLE.get(sort, SORTABLE["cost_usd"])
    listing = rows(con.execute(
        TRAJECTORY_SELECT + where +
        " ORDER BY %s %s NULLS LAST, t.trajectory_id LIMIT ? OFFSET ?" % (
            sort_sql, "DESC" if order == "desc" else "ASC"),
        params + [limit, offset]))
    summary = rows(con.execute("""
        SELECT count(*) AS trajectories, sum(t.n_calls) AS calls,
               sum(t.logged_total_cost_usd) AS cost_usd,
               sum(t.routed_cost_usd) AS routed_cost_usd,
               avg(m.cache_hit_rate) AS cache_hit_rate
        FROM trajectories t JOIN trajectory_meta m USING(trajectory_id)""" + where,
        params))[0]
    facets = {}
    for key, column in FILTERABLE.items():
        facets[key] = rows(con.execute("""
            SELECT coalesce(CAST(%s AS VARCHAR), '(none)') AS value, count(*) AS n,
                   sum(t.logged_total_cost_usd) AS cost_usd
            FROM trajectories t JOIN trajectory_meta m USING(trajectory_id)%s
            GROUP BY 1 ORDER BY cost_usd DESC LIMIT 40""" % (column, where), params))
    return {"rows": listing, "summary": summary, "facets": facets,
            "sort": sort, "order": order, "limit": limit, "offset": offset}


def _text_of(item: Dict[str, Any], budget: int = 900) -> str:
    """Readable one-screen rendering of a single request item."""
    if not isinstance(item, dict):
        return json.dumps(item)[:budget]
    kind = item.get("type")
    if kind in ("function_call", "custom_tool_call"):
        payload = item.get("arguments") or item.get("input") or ""
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return payload[:budget]
    if kind in ("function_call_output", "custom_tool_call_output"):
        payload = item.get("output")
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return payload[:budget]
    if kind == "reasoning":
        summary = item.get("summary") or []
        return json.dumps(summary, ensure_ascii=False)[:budget]
    content = item.get("content")
    if isinstance(content, str):
        return content[:budget]
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("input_text", "output_text", "text"):
                parts.append(part.get("text") or "")
            elif part.get("type") == "input_image":
                parts.append("[image]")
        return "\n".join(parts)[:budget]
    return json.dumps(item, ensure_ascii=False)[:budget]


def trajectory_detail(con: Any, trajectory_id: str) -> Dict[str, Any]:
    header = rows(con.execute(
        TRAJECTORY_SELECT + " WHERE t.trajectory_id = ?", [trajectory_id]))
    if not header:
        return {}
    calls = rows(con.execute("""
        SELECT call_id, call_index, is_final, model, est_input_tokens, tools_tokens,
               est_new_tokens, shared_prefix_tokens, prefix_share, est_output_tokens,
               n_items, last_item_type, last_tool_name, output_kind,
               output_n_tool_calls, output_tool_names, output_prose_tokens,
               tool_output_error, repeat_tool, has_image,
               logged_input_cost_usd, logged_output_cost_usd, logged_total_cost_usd,
               chosen_model, direction, cost_usd AS routed_cost_usd,
               switch_penalty_usd, raw_id, output_start_item, output_stop_item
        FROM calls WHERE trajectory_id = ? ORDER BY call_index""", [trajectory_id]))
    opening = rows(con.execute("""
        SELECT o.opening_user, o.system_prompt FROM raw_openings o
        JOIN calls c ON c.raw_id = o.raw_id WHERE c.trajectory_id = ? LIMIT 1""",
        [trajectory_id]))
    raw_id = calls[0]["raw_id"] if calls else None
    items: List[Dict[str, Any]] = []
    if raw_id is not None:
        record = con.execute(
            "SELECT raw_json FROM raw_lines WHERE raw_id = ?", [raw_id]).fetchone()
        if record:
            row = json.loads(record[0])
            # Which call each item belongs to: item i is part of call k's OUTPUT when it
            # falls in that call's recovered span, otherwise it is environment input
            # accumulating into the next call's prompt.
            spans = {}
            for call in calls:
                start, stop = call["output_start_item"], call["output_stop_item"]
                if start is None or stop is None:
                    continue
                for index in range(start, stop):
                    spans[index] = call["call_index"]
            boundaries = sorted(
                call["output_start_item"] for call in calls
                if call["output_start_item"] is not None)
            for index, item in enumerate(row.get("input") or []):
                belongs = spans.get(index)
                if belongs is None:
                    belongs = sum(1 for b in boundaries if b <= index)
                items.append({
                    "index": index,
                    "type": item.get("type") or item.get("role") or "message",
                    "role": item.get("role"),
                    "tool": item.get("name"),
                    "tokens": len(json.dumps(item)) // 4,
                    "call_index": belongs,
                    "is_output": index in spans,
                    "text": _text_of(item),
                })
    return {
        "trajectory": header[0], "calls": calls, "items": items,
        "opening_user": opening[0]["opening_user"] if opening else None,
        "system_prompt": opening[0]["system_prompt"] if opening else None,
    }


def _step_signature(call: Dict[str, Any]) -> str:
    """What a call DID, ignoring how big its prompt happened to be. Two calls with the
    same signature are the same move in the agent loop, so they can be aligned."""
    names = call.get("output_tool_names") or []
    kind = call.get("output_kind") or "unobservable"
    return "%s(%s)" % (kind, ",".join(sorted(names)))


def _align(left: Sequence[str], right: Sequence[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Longest-common-subsequence alignment of two step-signature sequences.

    Returns (left_index, right_index) pairs; a None on either side is a step that
    only one of the two trajectories took.
    """
    n, m = len(left), len(right)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (table[i + 1][j + 1] + 1) if left[i] == right[j] else max(
                table[i + 1][j], table[i][j + 1])
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i = j = 0
    while i < n and j < m:
        if left[i] == right[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            pairs.append((i, None)); i += 1
        else:
            pairs.append((None, j)); j += 1
    pairs.extend((i + k, None) for k in range(n - i))
    pairs.extend((None, j + k) for k in range(m - j))
    return pairs


def compare(con: Any, left_id: str, right_id: str) -> Dict[str, Any]:
    """Align two trajectories step by step and attribute the cost difference.

    The alignment is STRUCTURAL, not textual: every export row opens with a unique
    user message, so two trajectories never share a literal prompt prefix. What they
    do share is the shape of the loop — the same tool called at the same point. The
    matched steps are the comparable part; the cost gap across them is the part a
    router could actually act on.
    """
    sides = []
    for trajectory_id in (left_id, right_id):
        detail = trajectory_detail(con, trajectory_id)
        if not detail:
            return {}
        sides.append(detail)
    left, right = sides
    left_steps = [_step_signature(call) for call in left["calls"]]
    right_steps = [_step_signature(call) for call in right["calls"]]
    pairs = _align(left_steps, right_steps)

    steps, matched_left, matched_right = [], 0.0, 0.0
    for left_index, right_index in pairs:
        left_call = left["calls"][left_index] if left_index is not None else None
        right_call = right["calls"][right_index] if right_index is not None else None
        status = "matched" if left_call and right_call else (
            "only_left" if left_call else "only_right")
        if status == "matched":
            matched_left += left_call["logged_total_cost_usd"] or 0.0
            matched_right += right_call["logged_total_cost_usd"] or 0.0
        steps.append({
            "status": status,
            "signature": _step_signature(left_call or right_call),
            "left": _step_view(left_call), "right": _step_view(right_call),
        })
    left_total = left["trajectory"]["cost_usd"] or 0.0
    right_total = right["trajectory"]["cost_usd"] or 0.0
    union = len(set(left_steps) | set(right_steps))
    return {
        "left": {"trajectory": left["trajectory"], "steps": left_steps},
        "right": {"trajectory": right["trajectory"], "steps": right_steps},
        "steps": steps,
        "summary": {
            "matched_steps": sum(1 for step in steps if step["status"] == "matched"),
            "total_steps": len(steps),
            "shape_overlap": (len(set(left_steps) & set(right_steps)) / union) if union else None,
            "matched_cost_left": matched_left, "matched_cost_right": matched_right,
            "matched_cost_delta": matched_right - matched_left,
            "total_cost_left": left_total, "total_cost_right": right_total,
            "total_cost_delta": right_total - left_total,
            "unmatched_cost_left": left_total - matched_left,
            "unmatched_cost_right": right_total - matched_right,
            "same_model": left["trajectory"]["model"] == right["trajectory"]["model"],
        },
    }


def _step_view(call: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if call is None:
        return None
    return {
        "call_index": call["call_index"], "model": call["model"],
        "output_kind": call["output_kind"], "tools": call["output_tool_names"],
        "est_input_tokens": call["est_input_tokens"],
        "est_new_tokens": call["est_new_tokens"],
        "est_output_tokens": call["est_output_tokens"],
        "prefix_share": call["prefix_share"],
        "cost_usd": call["logged_total_cost_usd"],
        "tool_output_error": call["tool_output_error"],
    }


def similar(con: Any, trajectory_id: str, limit: int = 12) -> Dict[str, Any]:
    """Nearest trajectories by loop shape and tool vocabulary.

    Scored on three things a router could see BEFORE the run: the toolset it was
    handed, the workload class, and the sequence of moves the loop made. Ranking
    by these and reading off the cost gap is the crudest possible off-policy
    comparison — it is only as good as the assumption that same-shape runs are
    exchangeable, which is exactly where it can fail.
    """
    anchor = rows(con.execute(
        TRAJECTORY_SELECT + " WHERE t.trajectory_id = ?", [trajectory_id]))
    if not anchor:
        return {}
    anchor = anchor[0]
    tools = set((anchor["tool_signature"] or "").split(",")) - {""}
    candidates = rows(con.execute(
        TRAJECTORY_SELECT + " WHERE t.trajectory_id <> ?", [trajectory_id]))
    shape = anchor["shape_signature"] or ""
    scored = []
    for row in candidates:
        other = set((row["tool_signature"] or "").split(",")) - {""}
        union = tools | other
        jaccard = (len(tools & other) / len(union)) if union else 0.0
        same_shape = 1.0 if row["shape_signature"] == shape else 0.0
        depth = 1.0 - min(abs((row["n_calls"] or 0) - (anchor["n_calls"] or 0)) /
                          max(anchor["n_calls"] or 1, 1), 1.0)
        score = 0.45 * jaccard + 0.30 * same_shape + 0.25 * depth
        row["similarity"] = score
        row["tool_jaccard"] = jaccard
        row["same_shape"] = bool(same_shape)
        row["cost_delta_usd"] = (row["cost_usd"] or 0) - (anchor["cost_usd"] or 0)
        scored.append(row)
    scored.sort(key=lambda r: -r["similarity"])
    return {"anchor": anchor, "neighbours": scored[:limit]}


def matched_cohorts(con: Any, axes: Sequence[str], min_models: int = 2) -> Dict[str, Any]:
    """The off-policy estimate, done the honest way: match, then compare.

    Trajectories are bucketed on axes a router could read BEFORE the first token
    (toolset, workload, trigger, size). Within a bucket the model that served the
    run is treated as if it were assigned at random, so the per-model mean cost in
    that bucket estimates what the OTHER runs in it would have cost on that model.
    Buckets that only ever saw one model carry no counterfactual and are excluded.

    Where this fails, and it is not a small caveat: assignment was NOT random. If
    the dispatcher already sent hard work to expensive models, then within a bucket
    the expensive model's runs are the hard ones, and this estimator reads that
    selection as if it were a price difference. Confounder count per bucket is
    reported so the reader can judge how thin the matching is.
    """
    axes = [axis for axis in axes if axis in FILTERABLE] or ["toolset_label", "workload", "size_band"]
    key_sql = ", ".join(FILTERABLE[axis] for axis in axes)
    label_sql = " || ' · ' || ".join(
        "coalesce(CAST(%s AS VARCHAR), '(none)')" % FILTERABLE[axis] for axis in axes)
    cells = rows(con.execute("""
        WITH cell AS (
          SELECT %s AS cohort, t.model,
                 t.logged_total_cost_usd AS cost_usd, t.n_calls,
                 m.billed_input_tokens
          FROM trajectories t JOIN trajectory_meta m USING(trajectory_id)
          WHERE t.logged_total_cost_usd IS NOT NULL)
        SELECT cohort, model, count(*) AS trajectories,
               avg(cost_usd) AS mean_cost_usd, median(cost_usd) AS median_cost_usd,
               sum(cost_usd) AS cost_usd, avg(n_calls) AS mean_calls,
               avg(billed_input_tokens) AS mean_billed_input_tokens
        FROM cell GROUP BY cohort, model ORDER BY cohort, mean_cost_usd""" % label_sql))
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in cells:
        bucket = grouped.setdefault(row["cohort"], {"cohort": row["cohort"], "models": []})
        bucket["models"].append(row)
    cohorts = []
    for bucket in grouped.values():
        models = bucket["models"]
        if len(models) < min_models:
            continue
        cheapest = min(models, key=lambda r: r["mean_cost_usd"])
        total = sum(row["trajectories"] for row in models)
        actual = sum(row["cost_usd"] for row in models)
        counterfactual = cheapest["mean_cost_usd"] * total
        # How comparable are the cells really? Wide spread in prompt size inside a
        # cohort means the models were NOT seeing the same work.
        sizes = [row["mean_billed_input_tokens"] or 0 for row in models]
        spread = (max(sizes) / min(sizes)) if min(sizes) else None
        cohorts.append({
            **bucket, "trajectories": total, "distinct_models": len(models),
            "actual_cost_usd": actual, "cheapest_model": cheapest["model"],
            "counterfactual_cost_usd": counterfactual,
            "saving_usd": actual - counterfactual,
            "saving_pct": (actual - counterfactual) / actual if actual else None,
            "size_spread": spread,
        })
    cohorts.sort(key=lambda c: -(c["saving_usd"] or 0))
    covered = sum(c["trajectories"] for c in cohorts)
    total_trajectories = con.execute("SELECT count(*) FROM trajectories").fetchone()[0]
    return {
        "axes": axes, "cohorts": cohorts,
        "coverage": {
            "matched_trajectories": covered, "total_trajectories": total_trajectories,
            "share": covered / total_trajectories if total_trajectories else None,
            "total_saving_usd": sum(c["saving_usd"] or 0 for c in cohorts),
            "matched_actual_usd": sum(c["actual_cost_usd"] or 0 for c in cohorts),
        },
    }


def source_files(con: Any) -> Dict[str, Any]:
    """The files on disk, with what each one actually contributed to the index.

    `export` rows are trajectories: one line unfolds into many calls. `results` rows
    are a router's decisions, which only attach if their trajectory ids were minted
    the same way — see route_join().
    """
    files = rows(con.execute("""
        SELECT s.kind, s.path, s.line_count, s.parse_errors, s.size_bytes, s.mtime_ns
        FROM sources s ORDER BY s.kind, s.path"""))
    rollup = {
        (row["source_path"]): row for row in rows(con.execute("""
            SELECT source_path, count(*) AS calls,
                   count(DISTINCT trajectory_id) AS trajectories,
                   count(DISTINCT model) AS models,
                   sum(est_input_tokens + coalesce(tools_tokens, 0)) AS billed_input_tokens,
                   sum(est_output_tokens) AS output_tokens,
                   sum(logged_total_cost_usd) AS cost_usd
            FROM calls GROUP BY source_path"""))
    }
    ledger = {
        row["source_path"]: row for row in rows(con.execute("""
            SELECT x.source_path, count(*) AS entries,
                   count(DISTINCT l.trajectory_id) AS trajectory_ids
            FROM ledger_entries l JOIN raw_lines x ON x.raw_id = l.ledger_raw_id
            GROUP BY x.source_path"""))
    }
    errors = {
        row["source_path"]: row["messages"] for row in rows(con.execute("""
            SELECT source_path, list(message)[1:3] AS messages
            FROM index_errors GROUP BY source_path"""))
    }
    for row in files:
        row["contributed"] = rollup.get(row["path"])
        row["router_entries"] = ledger.get(row["path"])
        row["error_samples"] = errors.get(row["path"]) or []
    return {"files": files}


def source_rows(
    con: Any, path: str, offset: int = 0, limit: int = 100,
) -> Dict[str, Any]:
    """One page of raw lines from a single file, each with what it decomposed into."""
    kind = con.execute(
        "SELECT kind FROM sources WHERE path = ?", [path]).fetchone()
    if not kind:
        return {}
    kind = kind[0]
    total = con.execute(
        "SELECT count(*) FROM raw_lines WHERE source_path = ?", [path]).fetchone()[0]
    if kind == "export":
        listing = rows(con.execute("""
            SELECT x.source_line, x.raw_id, length(x.raw_json) AS raw_chars,
                   e.n_items,
                   any_value(c.trajectory_id) AS trajectory_id,
                   any_value(c.model) AS model,
                   count(c.call_id) AS calls,
                   sum(c.est_input_tokens + coalesce(c.tools_tokens, 0)) AS billed_input_tokens,
                   sum(c.est_output_tokens) AS output_tokens,
                   sum(c.logged_total_cost_usd) AS cost_usd,
                   any_value(len(c.tools_available)) AS n_tools
            FROM raw_lines x
            LEFT JOIN export_rows e USING(raw_id)
            LEFT JOIN calls c ON c.raw_id = x.raw_id
            WHERE x.source_path = ?
            GROUP BY x.source_line, x.raw_id, raw_chars, e.n_items
            ORDER BY x.source_line LIMIT ? OFFSET ?""", [path, limit, offset]))
    else:
        listing = rows(con.execute("""
            SELECT x.source_line, x.raw_id, length(x.raw_json) AS raw_chars,
                   any_value(l.trajectory_id) AS trajectory_id,
                   any_value(l.logged_model) AS model,
                   count(l.call_index) AS calls,
                   count(t.trajectory_id) > 0 AS joins_to_export
            FROM raw_lines x
            LEFT JOIN ledger_entries l ON l.ledger_raw_id = x.raw_id
            LEFT JOIN trajectories t ON t.trajectory_id = l.trajectory_id
            WHERE x.source_path = ?
            GROUP BY x.source_line, x.raw_id, raw_chars
            ORDER BY x.source_line LIMIT ? OFFSET ?""", [path, limit, offset]))
    return {"path": path, "kind": kind, "total": total, "offset": offset,
            "limit": limit, "rows": listing}


RAW_PREVIEW_CHARS = 24_000


def source_row(con: Any, path: str, line: int) -> Dict[str, Any]:
    """One raw line, pretty-printed. Large rows are truncated on purpose: an export
    row can be several hundred KB, and the readable view of it is the Flow tab."""
    record = con.execute(
        "SELECT raw_id, raw_json FROM raw_lines WHERE source_path = ? AND source_line = ?",
        [path, line]).fetchone()
    if not record:
        return {}
    raw_id, raw_json = record
    parsed: Optional[Dict[str, Any]]
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed = None
    shape: Dict[str, Any] = {"keys": sorted(parsed) if isinstance(parsed, dict) else None}
    if isinstance(parsed, dict):
        shape["model"] = parsed.get("model")
        shape["n_input_items"] = len(parsed.get("input") or [])
        shape["tools"] = [
            tool.get("name") or (tool.get("function") or {}).get("name")
            for tool in (parsed.get("tools") or []) if isinstance(tool, dict)
        ]
    pretty = json.dumps(parsed, indent=2, ensure_ascii=False) if parsed is not None else raw_json
    trajectory = con.execute(
        "SELECT trajectory_id FROM calls WHERE raw_id = ? LIMIT 1", [raw_id]).fetchone()
    return {
        "path": path, "line": line, "raw_id": raw_id, "shape": shape,
        "trajectory_id": trajectory[0] if trajectory else None,
        "total_chars": len(pretty), "truncated": len(pretty) > RAW_PREVIEW_CHARS,
        "preview": pretty[:RAW_PREVIEW_CHARS],
    }


# The one comparison the export genuinely supports: hold the FUNCTION fixed and let the
# model vary. Same tool, same kind of command — what did each model actually emit, and
# what did that turn cost? It is still observational (nobody assigned these at random),
# but the unit of comparison is a concrete, readable thing rather than a whole run.
FUNCTION_MODEL_STATS = """
    count(*) AS calls,
    count(DISTINCT f.trajectory_id) AS trajectories,
    avg(f.arg_tokens) AS mean_arg_tokens,
    median(f.arg_tokens) AS median_arg_tokens,
    avg(f.result_tokens) AS mean_result_tokens,
    median(f.result_tokens) AS median_result_tokens,
    avg(CAST(f.result_error AS DOUBLE)) AS error_rate,
    sum(f.cost_share_usd) AS spend_usd,
    avg(f.cost_share_usd) AS mean_cost_usd,
    median(f.cost_share_usd) AS median_cost_usd,
    avg(f.request_input_tokens) AS mean_request_input_tokens,
    any_value(p.uncached) AS price_uncached,
    any_value(p.output) AS price_output
"""
FUNCTION_FROM = "FROM function_calls f LEFT JOIN model_prices p ON p.model = f.model"


def function_groups(con: Any) -> Dict[str, Any]:
    """Every tool, with the models that called it and what each one paid per call."""
    tools = rows(con.execute("""
        SELECT tool, count(DISTINCT f.model) AS models, %s
        %s GROUP BY tool ORDER BY calls DESC""" % (FUNCTION_MODEL_STATS, FUNCTION_FROM)))
    by_model = rows(con.execute("""
        SELECT tool, f.model, model_family, %s
        %s GROUP BY tool, f.model, model_family
        ORDER BY tool, calls DESC""" % (FUNCTION_MODEL_STATS, FUNCTION_FROM)))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in by_model:
        grouped.setdefault(row["tool"], []).append(row)
    for tool in tools:
        tool["by_model"] = grouped.get(tool["tool"], [])
        costs = [row["mean_cost_usd"] for row in tool["by_model"] if row["mean_cost_usd"]]
        tool["cost_spread"] = (max(costs) / min(costs)) if costs and min(costs) else None
    total = con.execute("SELECT count(*), sum(cost_share_usd) FROM function_calls").fetchone()
    return {
        "tools": tools,
        "totals": {"calls": total[0], "spend_usd": total[1]},
    }


def function_detail(
    con: Any, tool: str, group_by: str = "command_verb", limit: int = 40,
) -> Dict[str, Any]:
    """One tool, split into comparable groups, each broken down by model.

    `command_verb` groups shell calls by the executable they ran — the closest the
    export gets to "the same job done twice". `arg_shape` groups by which argument
    keys were set, which is what the non-shell tools vary on.
    """
    if group_by not in ("command_verb", "arg_shape"):
        group_by = "command_verb"
    exists = con.execute(
        "SELECT count(*) FROM function_calls WHERE tool = ?", [tool]).fetchone()[0]
    if not exists:
        return {}
    overall = rows(con.execute("""
        SELECT f.model, model_family, %s %s WHERE tool = ?
        GROUP BY f.model, model_family ORDER BY calls DESC"""
        % (FUNCTION_MODEL_STATS, FUNCTION_FROM), [tool]))
    groups = rows(con.execute("""
        SELECT coalesce(f.%s, '(none)') AS group_key,
               count(DISTINCT f.model) AS models, %s
        %s WHERE tool = ?
        GROUP BY 1 ORDER BY calls DESC LIMIT ?"""
        % (group_by, FUNCTION_MODEL_STATS, FUNCTION_FROM), [tool, limit]))
    group_models = rows(con.execute("""
        SELECT coalesce(f.%s, '(none)') AS group_key, f.model, %s
        %s WHERE tool = ?
        GROUP BY 1, f.model ORDER BY group_key, calls DESC"""
        % (group_by, FUNCTION_MODEL_STATS, FUNCTION_FROM), [tool]))
    per_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in group_models:
        per_group.setdefault(row["group_key"], []).append(row)
    for group in groups:
        group["by_model"] = per_group.get(group["group_key"], [])
    return {
        "tool": tool, "group_by": group_by,
        "by_model": overall, "groups": groups,
        "comparable_groups": sum(1 for group in groups if group["models"] > 1),
    }


SAMPLE_CHARS = 1400


def function_samples(
    con: Any, tool: str, group_key: Optional[str] = None,
    group_by: str = "command_verb", per_model: int = 3,
) -> Dict[str, Any]:
    """Actual emitted arguments for one tool (and optionally one group), by model.

    Samples are the median-length call for each model plus its neighbours, not the
    first rows: the first rows of a trajectory are systematically the setup calls.
    """
    if group_by not in ("command_verb", "arg_shape"):
        group_by = "command_verb"
    clauses, params = ["tool = ?"], [tool]
    if group_key is not None:
        clauses.append("coalesce(%s, '(none)') = ?" % group_by)
        params.append(group_key)
    where = " WHERE " + " AND ".join(clauses)
    samples = rows(con.execute("""
        WITH scoped AS (
          SELECT fc_id, trajectory_id, call_index, model, arg_text, arg_tokens,
                 result_text, result_tokens, result_error, cost_share_usd,
                 request_cost_usd, n_calls_in_request,
                 row_number() OVER (PARTITION BY model ORDER BY arg_tokens) AS rank_asc,
                 count(*) OVER (PARTITION BY model) AS model_calls
          FROM function_calls%s
        )
        SELECT fc_id, trajectory_id, call_index, model, arg_tokens, result_tokens,
               result_error, cost_share_usd, request_cost_usd, n_calls_in_request,
               model_calls,
               substr(arg_text, 1, %d) AS arg_text,
               length(arg_text) AS arg_chars,
               substr(result_text, 1, %d) AS result_text,
               length(result_text) AS result_chars
        FROM scoped
        WHERE abs(rank_asc - (model_calls + 1) / 2) <= ?
        ORDER BY model, rank_asc""" % (where, SAMPLE_CHARS, SAMPLE_CHARS),
        params + [max(per_model - 1, 0) / 2.0 + 0.5]))
    by_model: Dict[str, Dict[str, Any]] = {}
    for row in samples:
        bucket = by_model.setdefault(row["model"], {
            "model": row["model"], "calls": row["model_calls"], "samples": [],
        })
        if len(bucket["samples"]) < per_model:
            bucket["samples"].append(row)
    return {
        "tool": tool, "group_key": group_key, "group_by": group_by,
        "models": sorted(by_model.values(), key=lambda entry: -entry["calls"]),
        "sample_chars": SAMPLE_CHARS,
    }
