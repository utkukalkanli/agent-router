"""Build the local DuckDB index used by the trajectory explorer.

Supports request-snapshot exports and complete-history exports. Source files
are read once per rebuild; untouched lines are stored in ``raw_lines`` and all
later work is done from DuckDB. Token counts are estimated JSON chars / 4.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse the repository's canonical token, cache, call-boundary, and ID logic.
from cost_model import load_pricing, price_of, shared_prefix_tokens  # noqa: E402
from decompose import decompose, is_model_item  # noqa: E402
from load_trajectories import est_tokens, first_user_text, group_key  # noqa: E402


DB_PATH = Path(__file__).with_name("index.duckdb")
INDEX_VERSION = "6"
Progress = Optional[Callable[[str], None]]
ERROR_RE = re.compile(
    r"(?:\berror\b|\bfailed\b|\bfailure\b|\bexception\b|traceback|"
    r"permission denied|not found|invalid request)", re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceFile:
    kind: str
    path: Path
    relative_path: str
    mtime_ns: int
    size_bytes: int


@dataclass(frozen=True)
class IndexResult:
    rebuilt: bool
    calls: int
    trajectories: int
    export_files: int
    result_files: int
    elapsed_seconds: float
    reason: str


CALL_COLUMNS = [
    "call_id", "trajectory_id", "call_index", "n_calls_in_trajectory",
    "is_final", "model", "model_family", "tier_rank", "est_input_tokens",
    "est_new_tokens", "shared_prefix_tokens", "prefix_share", "n_items",
    "item_type_sequence", "item_tokens", "last_item_type", "tools_available",
    "tools_used_so_far", "last_tool_name", "n_distinct_tools", "has_image",
    "has_reasoning", "tool_output_error", "repeat_tool", "output_kind",
    "output_n_tool_calls", "output_tool_names", "output_prose_tokens", "routine",
    "chosen_model", "kept_logged", "direction", "p_routine", "flip_lambda",
    "cost_usd", "switch_penalty_usd", "split", "source_kind", "source_path",
    "source_line", "raw_id", "delta_raw_id", "delta_start_item",
    "delta_stop_item", "output_raw_id", "output_start_item", "output_stop_item",
    "ledger_raw_id", "logged_input_cost_usd", "tools_tokens",
    "est_output_tokens", "logged_output_cost_usd", "logged_total_cost_usd",
]


def discover_sources(root: Path = ROOT) -> List[SourceFile]:
    """Return optional export/results JSONL files in deterministic order."""
    root = Path(root)
    found: List[SourceFile] = []
    for kind, directory in (("export", root / "export"), ("results", root / "results")):
        paths = sorted(directory.glob("*.jsonl")) if directory.exists() else []
        for path in paths:
            stat = path.stat()
            found.append(SourceFile(
                kind, path, str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size,
            ))
    return found


def _manifest(sources: Sequence[SourceFile]) -> List[Tuple[str, str, int, int]]:
    return [(s.kind, s.relative_path, s.mtime_ns, s.size_bytes) for s in sources]


def _database_is_current(db_path: Path, sources: Sequence[SourceFile]) -> bool:
    if not db_path.exists():
        return False
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        version = con.execute(
            "SELECT value FROM index_meta WHERE key = 'index_version'"
        ).fetchone()
        rows = con.execute(
            "SELECT kind, path, mtime_ns, size_bytes FROM sources ORDER BY kind, path"
        ).fetchall()
        con.close()
        return bool(version and version[0] == INDEX_VERSION and rows == _manifest(sources))
    except Exception:
        return False


def ensure_index(
    root: Path = ROOT,
    db_path: Path = DB_PATH,
    force: bool = False,
    progress: Progress = print,
) -> IndexResult:
    """Build only when a source mtime/size or index schema version changed."""
    started = time.monotonic()
    root, db_path = Path(root), Path(db_path)
    sources = discover_sources(root)
    if not force and _database_is_current(db_path, sources):
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        calls = con.execute("SELECT count(*) FROM calls").fetchone()[0]
        trajectories = con.execute("SELECT count(*) FROM trajectories").fetchone()[0]
        con.close()
        return IndexResult(
            False, calls, trajectories,
            sum(s.kind == "export" for s in sources),
            sum(s.kind == "results" for s in sources),
            time.monotonic() - started, "source manifest unchanged",
        )
    return _rebuild(root, db_path, sources, progress, started)


def build(root: Path = ROOT, db_path: Path = DB_PATH) -> Tuple[int, int]:
    """Backward-compatible force-build entry point."""
    result = ensure_index(root, db_path, force=True)
    return result.calls, result.trajectories


def _emit(progress: Progress, message: str) -> None:
    if progress:
        progress(message)


def _create_schema(con: Any) -> None:
    con.execute("CREATE TABLE index_meta(key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.execute("""CREATE TABLE sources(
        kind VARCHAR, path VARCHAR, mtime_ns BIGINT, size_bytes BIGINT,
        line_count BIGINT, parse_errors BIGINT, PRIMARY KEY(kind, path))""")
    con.execute("""CREATE TABLE raw_lines(
        raw_id BIGINT PRIMARY KEY, source_kind VARCHAR, source_path VARCHAR,
        source_line BIGINT, raw_json VARCHAR)""")
    con.execute("CREATE TABLE index_errors(source_path VARCHAR, source_line BIGINT, message VARCHAR)")
    con.execute("CREATE TABLE export_rows(raw_id BIGINT PRIMARY KEY, group_signature VARCHAR, n_items INTEGER)")
    con.execute("""CREATE TABLE raw_openings(
        raw_id BIGINT PRIMARY KEY, opening_user VARCHAR, system_prompt VARCHAR)""")
    con.execute("""CREATE TABLE raw_items(
        raw_id BIGINT, item_index INTEGER, item_type VARCHAR, item_tokens BIGINT,
        is_model_authored BOOLEAN, tool_name VARCHAR, search_field VARCHAR,
        search_text VARCHAR, PRIMARY KEY(raw_id, item_index))""")
    con.execute("""CREATE TABLE ledger_entries(
        trajectory_id VARCHAR, call_index INTEGER, priority BIGINT,
        chosen_model VARCHAR, kept_logged BOOLEAN, direction VARCHAR,
        p_routine DOUBLE, flip_lambda DOUBLE, cost_usd DOUBLE,
        switch_penalty_usd DOUBLE, split VARCHAR, routine BOOLEAN,
        logged_model VARCHAR, n_calls INTEGER, ledger_raw_id BIGINT)""")
    con.execute("""CREATE TABLE result_calls(
        trajectory_id VARCHAR, call_index INTEGER, n_calls INTEGER,
        model VARCHAR, model_family VARCHAR, est_input_tokens BIGINT,
        est_new_tokens BIGINT, shared_prefix_tokens BIGINT,
        output_kind VARCHAR, output_n_tool_calls INTEGER,
        output_tool_names VARCHAR[], output_prose_tokens BIGINT,
        routine BOOLEAN, raw_id BIGINT)""")
    con.execute("""CREATE TABLE result_trajectories(
        trajectory_id VARCHAR, model VARCHAR, model_family VARCHAR,
        n_calls INTEGER, raw_id BIGINT)""")
    con.execute("""CREATE TABLE calls(
        call_id VARCHAR PRIMARY KEY,
        trajectory_id VARCHAR NOT NULL,
        call_index INTEGER NOT NULL,
        n_calls_in_trajectory INTEGER,
        is_final BOOLEAN,
        model VARCHAR,
        model_family VARCHAR,
        tier_rank INTEGER,
        est_input_tokens BIGINT,
        est_new_tokens BIGINT,
        shared_prefix_tokens BIGINT,
        prefix_share DOUBLE,
        n_items INTEGER,
        item_type_sequence VARCHAR,
        item_tokens BIGINT[],
        last_item_type VARCHAR,
        tools_available VARCHAR[],
        tools_used_so_far VARCHAR[],
        last_tool_name VARCHAR,
        n_distinct_tools INTEGER,
        has_image BOOLEAN,
        has_reasoning BOOLEAN,
        tool_output_error BOOLEAN,
        repeat_tool BOOLEAN,
        output_kind VARCHAR,
        output_n_tool_calls INTEGER,
        output_tool_names VARCHAR[],
        output_prose_tokens BIGINT,
        routine BOOLEAN,
        chosen_model VARCHAR,
        kept_logged BOOLEAN,
        direction VARCHAR,
        p_routine DOUBLE,
        flip_lambda DOUBLE,
        cost_usd DOUBLE,
        switch_penalty_usd DOUBLE,
        split VARCHAR,
        source_kind VARCHAR,
        source_path VARCHAR,
        source_line BIGINT,
        raw_id BIGINT,
        delta_raw_id BIGINT,
        delta_start_item INTEGER,
        delta_stop_item INTEGER,
        output_raw_id BIGINT,
        output_start_item INTEGER,
        output_stop_item INTEGER,
        ledger_raw_id BIGINT,
        logged_input_cost_usd DOUBLE,
        tools_tokens BIGINT,
        est_output_tokens BIGINT,
        logged_output_cost_usd DOUBLE,
        logged_total_cost_usd DOUBLE)""")


def _opening_signature(row: Dict[str, Any]) -> str:
    items = row.get("input") or []
    system = next(
        (x for x in items if isinstance(x, dict) and x.get("role") == "system"), None,
    )
    value = json.dumps([system, first_user_text(row)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(value.encode()).hexdigest()


def _full_trajectory_id(items: Sequence[Dict[str, Any]]) -> str:
    """Match trajectory_ids.py and decompose.py exactly."""
    return hashlib.sha1(json.dumps(items, sort_keys=True).encode()).hexdigest()[:12]


def _model_family(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gpt"):
        return "gpt"
    return "other"


def _result_output_kind(row: Dict[str, Any]) -> Optional[str]:
    if row.get("is_final") is True:
        return "unobservable"
    if row.get("output_kind") is not None:
        return row["output_kind"]
    kinds = row.get("output_kinds")
    names = row.get("output_tool_names", row.get("tool_names")) or []
    if "apply_patch" in names:
        return "patch"
    n_calls = sum(k in ("function_call", "custom_tool_call") for k in (kinds or []))
    if n_calls > 1:
        return "multi_tool"
    if n_calls == 1:
        return "tool_call"
    if kinds and any(k in ("message", "assistant") for k in kinds):
        return "text"
    return "unknown" if kinds is not None else None


def _normalise_result(
    row: Dict[str, Any], raw_id: int,
) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]:
    """Map permissive result records to fixed ledger and call schemas."""
    trajectory_id = row.get("trajectory_id", row.get("trajectory"))
    if not trajectory_id:
        return [], [], None
    direct_index = row.get("call_index")
    route = row.get("route")
    indices: List[Tuple[int, Optional[str]]] = []
    if isinstance(route, list):
        indices.extend((i, model) for i, model in enumerate(route))
    elif direct_index is not None:
        indices.append((int(direct_index), row.get("chosen_model", row.get("routed_model"))))
    entries = []
    for call_index, routed_model in indices:
        entries.append((
            str(trajectory_id), call_index, raw_id, routed_model,
            row.get("kept_logged"), row.get("direction"), row.get("p_routine"),
            row.get("flip_lambda"), row.get("cost_usd", row.get("cost_usd_est")),
            row.get("switch_penalty_usd"), row.get("split"), row.get("routine"),
            row.get("logged_model", row.get("model")),
            row.get("n_calls", row.get("n_calls_in_trajectory")), raw_id,
        ))
    result_calls = []
    if direct_index is not None:
        names = row.get("output_tool_names", row.get("tool_names"))
        result_calls.append((
            str(trajectory_id), int(direct_index),
            row.get("n_calls", row.get("n_calls_in_trajectory")), row.get("model"),
            row.get("model_family", row.get("family")),
            row.get("est_input_tokens", row.get("input_tokens_est")),
            row.get("est_new_tokens", row.get("uncached_tokens_est")),
            row.get("shared_prefix_tokens", row.get("cached_tokens_est")),
            _result_output_kind(row),
            row.get("output_n_tool_calls", len(names) if names is not None else None),
            names, row.get("output_prose_tokens", row.get("output_tokens_est")),
            None if row.get("is_final") is True else row.get("routine"), raw_id,
        ))
    trajectory = (
        str(trajectory_id), row.get("logged_model", row.get("model")),
        row.get("model_family", row.get("family")),
        row.get("n_calls", row.get("n_calls_in_trajectory")), raw_id,
    )
    return entries, result_calls, trajectory


def _copy_tuples(
    con: Any, table: str, columns: Sequence[str], rows: List[Tuple[Any, ...]],
) -> None:
    """Bulk-load Python tuples through DuckDB's typed JSON COPY path."""
    if not rows:
        return
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False,
    )
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(
                    dict(zip(columns, row)), ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
        con.execute("COPY %s FROM ? (FORMAT JSON)" % table, [handle.name])
    finally:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        rows.clear()


def _raw_item_tuple(raw_id: int, item_index: int, item: Any) -> Tuple[Any, ...]:
    item = item if isinstance(item, dict) else {"value": item}
    item_type = item.get("type")
    search_field = search_text = None
    if item_type in ("function_call", "custom_tool_call"):
        # function_call carries `arguments`; custom_tool_call (apply_patch) carries `input`.
        value = item.get("arguments", item.get("input"))
        if value is not None:
            search_field = "tool_arguments"
            search_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    elif item_type in ("function_call_output", "custom_tool_call_output"):
        search_field = "tool_outputs"
        value = item.get("output", item.get("content"))
        search_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (
        raw_id, item_index, _item_type(item), est_tokens(item), is_model_item(item),
        _tool_name(item), search_field, search_text,
    )


def _ingest_sources(con: Any, sources: Sequence[SourceFile], progress: Progress) -> None:
    raw_id = 0
    export_batch: List[Tuple[Any, ...]] = []
    opening_batch: List[Tuple[Any, ...]] = []
    item_batch: List[Tuple[Any, ...]] = []
    ledger_batch: List[Tuple[Any, ...]] = []
    result_call_batch: List[Tuple[Any, ...]] = []
    trajectory_batch: List[Tuple[Any, ...]] = []
    error_batch: List[Tuple[Any, ...]] = []

    def flush() -> None:
        _copy_tuples(con, "export_rows", ("raw_id", "group_signature", "n_items"), export_batch)
        _copy_tuples(con, "raw_openings", (
            "raw_id", "opening_user", "system_prompt",
        ), opening_batch)
        _copy_tuples(con, "raw_items", (
            "raw_id", "item_index", "item_type", "item_tokens",
            "is_model_authored", "tool_name", "search_field", "search_text",
        ), item_batch)
        _copy_tuples(con, "ledger_entries", (
            "trajectory_id", "call_index", "priority", "chosen_model",
            "kept_logged", "direction", "p_routine", "flip_lambda", "cost_usd",
            "switch_penalty_usd", "split", "routine", "logged_model", "n_calls",
            "ledger_raw_id",
        ), ledger_batch)
        _copy_tuples(con, "result_calls", (
            "trajectory_id", "call_index", "n_calls", "model", "model_family",
            "est_input_tokens", "est_new_tokens", "shared_prefix_tokens",
            "output_kind", "output_n_tool_calls", "output_tool_names",
            "output_prose_tokens", "routine", "raw_id",
        ), result_call_batch)
        _copy_tuples(con, "result_trajectories", (
            "trajectory_id", "model", "model_family", "n_calls", "raw_id",
        ), trajectory_batch)
        _copy_tuples(con, "index_errors", (
            "source_path", "source_line", "message",
        ), error_batch)

    for source in sources:
        _emit(progress, "Reading %s" % source.relative_path)
        parse_errors = 0
        first_raw_id = raw_id
        # A control character that JSON must escape makes JSONL a safe one-column
        # CSV. DuckDB performs this large copy much faster than Python parameter
        # binding, while retaining each source line byte-for-byte (minus newline).
        if source.size_bytes:
            con.execute("""INSERT INTO raw_lines
                WITH lines AS (
                  SELECT row_number() OVER () AS line_number, raw_json
                  FROM read_csv(?, delim='\\x1e', header=false,
                    columns={'raw_json':'VARCHAR'}, quote='', escape='',
                    strict_mode=false, parallel=false)
                )
                SELECT ? + line_number, ?, ?, line_number, raw_json FROM lines""",
                [str(source.path), first_raw_id, source.kind, source.relative_path],
            )
        line_count = con.execute(
            "SELECT count(*) FROM raw_lines WHERE source_path = ?", [source.relative_path],
        ).fetchone()[0]
        raw_id += line_count

        # cursor() creates an independent DuckDB connection, so writes performed
        # by flush() do not invalidate this streaming read result.
        reader = con.cursor()
        reader.execute("""SELECT raw_id, source_line, raw_json FROM raw_lines
            WHERE source_path = ? ORDER BY source_line""", [source.relative_path])
        processed = 0
        while True:
            chunk = reader.fetchmany(2_000)
            if not chunk:
                break
            for line_raw_id, line_number, raw in chunk:
                processed += 1
                try:
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError("top-level JSON value is not an object")
                    if source.kind == "export":
                        items = value.get("input")
                        if not isinstance(items, list):
                            raise ValueError("export row has no input item list")
                        export_batch.append((line_raw_id, _opening_signature(value), len(items)))
                        opening_batch.append((
                            line_raw_id, first_user_text(value), _system_prompt(items),
                        ))
                        item_batch.extend(
                            _raw_item_tuple(line_raw_id, index, item)
                            for index, item in enumerate(items)
                        )
                    else:
                        entries, result_calls, trajectory = _normalise_result(value, line_raw_id)
                        ledger_batch.extend(entries)
                        result_call_batch.extend(result_calls)
                        if trajectory:
                            trajectory_batch.append(trajectory)
                except Exception as exc:
                    parse_errors += 1
                    error_batch.append((source.relative_path, line_number, str(exc)))
            flush()
            if processed % 25_000 == 0:
                _emit(progress, "  {}: {:,d} lines".format(source.relative_path, processed))
        reader.close()
        flush()
        con.execute(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
            [source.kind, source.relative_path, source.mtime_ns, source.size_bytes,
             line_count, parse_errors],
        )


def _item_type(item: Any) -> str:
    if not isinstance(item, dict):
        return "unknown"
    item_type, role = item.get("type"), item.get("role")
    if item_type == "message" or (item_type is None and role):
        return {"system": "sys", "user": "user", "assistant": "assistant"}.get(
            role, role or "message",
        )
    return {
        "function_call": "fc", "function_call_output": "fco",
        "custom_tool_call": "ctc", "custom_tool_call_output": "ctco",
        "reasoning": "reasoning",
    }.get(item_type, item_type or "unknown")


def _tool_name(item: Dict[str, Any]) -> Optional[str]:
    if item.get("type") not in ("function_call", "custom_tool_call"):
        return None
    return item.get("name") or (item.get("function") or {}).get("name")


def _tool_definition_name(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    return tool.get("name") or (tool.get("function") or {}).get("name")


def _ordered_unique(values: Iterable[Optional[str]]) -> List[str]:
    seen, answer = set(), []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            answer.append(value)
    return answer


def _has_image(items: Sequence[Dict[str, Any]]) -> bool:
    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("input_image", "image_url")
            for part in content
        ):
            return True
    return False


def _last_tool_result(items: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in reversed(items):
        if isinstance(item, dict) and item.get("type") in (
            "function_call_output", "custom_tool_call_output",
        ):
            return item
    return None


def _text_content(item: Dict[str, Any]) -> str:
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


def _system_prompt(items: Sequence[Dict[str, Any]]) -> str:
    for item in items:
        if isinstance(item, dict) and item.get("role") == "system":
            return _text_content(item)
    return ""


def _output_metadata(
    model_items: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[Any, Any, Any, Any]:
    if model_items is None:
        return "unobservable", None, None, None
    calls = [item for item in model_items if _tool_name(item)]
    names = [_tool_name(item) for item in calls]
    prose_items = [
        item for item in model_items
        if (item.get("type") == "message" or item.get("type") is None)
        and item.get("role") == "assistant"
    ]
    prose_tokens = sum(est_tokens(_text_content(item)) for item in prose_items)
    if "apply_patch" in names:
        kind = "patch"
    elif len(calls) > 1:
        kind = "multi_tool"
    elif calls:
        kind = "tool_call"
    elif prose_items:
        kind = "text"
    else:
        kind = "unknown"
    return kind, len(calls), names, prose_tokens


def _shared_tokens(previous: Sequence[Dict[str, Any]], current: Sequence[Dict[str, Any]]) -> int:
    return shared_prefix_tokens({"input": previous}, {"input": current})


def _common_prefix_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def _tier_ranks(pricing: Dict[str, Sequence[float]]) -> Dict[str, int]:
    canonical = {
        "claude-opus-5", "claude-opus-4-8", "claude-opus-4-6",
        "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5",
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    }
    known = {model: values[0] for model, values in pricing.items() if model in canonical}
    prices = {price: rank for rank, price in enumerate(sorted(set(known.values())), 1)}
    return {model: prices[price] for model, price in known.items()}


def _base_call(
    trajectory_id: str, call_index: int, n_calls: int, row: Dict[str, Any],
    raw_id: int, source_path: str, source_line: int,
    input_items: Sequence[Dict[str, Any]],
    previous_input: Optional[Sequence[Dict[str, Any]]],
    output_items: Optional[Sequence[Dict[str, Any]]],
    delta_location: Optional[Tuple[int, int, int]],
    output_location: Optional[Tuple[int, int, int]],
    pricing: Dict[str, Sequence[float]], tier_ranks: Dict[str, int],
) -> Dict[str, Any]:
    model = row.get("model")
    types = [_item_type(item) for item in input_items]
    tool_history = [
        _tool_name(item) for item in input_items if isinstance(item, dict)
    ]
    tool_history = [name for name in tool_history if name]
    distinct_tools = _ordered_unique(tool_history)
    available = _ordered_unique(
        _tool_definition_name(tool) for tool in (row.get("tools") or [])
    )
    input_tokens = est_tokens(list(input_items))
    previous_tokens = est_tokens(list(previous_input)) if previous_input is not None else 0
    shared = _shared_tokens(previous_input or [], input_items) if previous_input is not None else 0
    last_result = _last_tool_result(input_items)
    output_kind, output_count, output_names, output_prose = _output_metadata(output_items)
    pu, pc, po = price_of(model, pricing)
    # Tool schemas are resent on every request, so they are billed input. They sit
    # in the cached prefix from the second call onward, exactly like the prompt items.
    tools_tokens = est_tokens(row.get("tools") or [])
    billed_input = input_tokens + tools_tokens
    cached = min(shared + tools_tokens, billed_input) if previous_input is not None else 0
    logged_cost = ((billed_input - cached) * pu + cached * pc) / 1e6
    # The export has no `usage`, but call k's output is recoverable: it is sitting in
    # call k+1's input. Only the final call's output is truly gone (None here).
    output_tokens = est_tokens(list(output_items)) if output_items is not None else None
    output_cost = (output_tokens * po / 1e6) if output_tokens is not None else None
    return {
        "call_id": "%s:%d" % (trajectory_id, call_index),
        "trajectory_id": trajectory_id, "call_index": call_index,
        "n_calls_in_trajectory": n_calls, "is_final": call_index == n_calls - 1,
        "model": model, "model_family": _model_family(model),
        "tier_rank": tier_ranks.get(model), "est_input_tokens": input_tokens,
        "est_new_tokens": input_tokens - previous_tokens if previous_input is not None else input_tokens,
        "shared_prefix_tokens": shared,
        "prefix_share": shared / input_tokens if input_tokens else None,
        "n_items": len(input_items), "item_type_sequence": ",".join(types),
        "item_tokens": [est_tokens(item) for item in input_items],
        "last_item_type": types[-1] if types else None,
        "tools_available": available, "tools_used_so_far": distinct_tools,
        "last_tool_name": tool_history[-1] if tool_history else None,
        "n_distinct_tools": len(distinct_tools), "has_image": _has_image(input_items),
        "has_reasoning": any(
            item.get("type") == "reasoning" for item in input_items
            if isinstance(item, dict)
        ),
        "tool_output_error": bool(
            last_result and ERROR_RE.search(json.dumps(last_result, ensure_ascii=False))
        ),
        "repeat_tool": len(tool_history) != len(set(tool_history)),
        "output_kind": output_kind, "output_n_tool_calls": output_count,
        "output_tool_names": output_names, "output_prose_tokens": output_prose,
        "routine": None, "chosen_model": None, "kept_logged": None,
        "direction": None, "p_routine": None, "flip_lambda": None,
        "cost_usd": None, "switch_penalty_usd": None, "split": None,
        "source_kind": "export", "source_path": source_path,
        "source_line": source_line, "raw_id": raw_id,
        "delta_raw_id": delta_location[0] if delta_location else None,
        "delta_start_item": delta_location[1] if delta_location else None,
        "delta_stop_item": delta_location[2] if delta_location else None,
        "output_raw_id": output_location[0] if output_location else None,
        "output_start_item": output_location[1] if output_location else None,
        "output_stop_item": output_location[2] if output_location else None,
        "ledger_raw_id": None, "logged_input_cost_usd": logged_cost,
        "tools_tokens": tools_tokens, "est_output_tokens": output_tokens,
        "logged_output_cost_usd": output_cost,
        "logged_total_cost_usd": logged_cost + (output_cost or 0.0),
    }


def _insert_call_batch(con: Any, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # DuckDB's Python executemany pays a large conversion cost for ~50 columns
    # and nested lists. Its native newline-delimited JSON COPY path is orders of
    # magnitude faster and casts by the already-declared target schema.
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False,
    )
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        con.execute("COPY calls FROM ? (FORMAT JSON)", [handle.name])
    finally:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        rows.clear()


def _snapshot_calls(
    records: Sequence[Tuple[int, str, int, Dict[str, Any]]],
    pricing: Dict[str, Sequence[float]], tier_ranks: Dict[str, int],
) -> Iterator[Dict[str, Any]]:
    ordered = sorted(records, key=lambda record: (len(record[3].get("input") or []), record[0]))
    trajectory_id, n_calls = group_key(ordered[0][3]), len(ordered)
    for index, (raw_id, source_path, source_line, row) in enumerate(ordered):
        items = row.get("input") or []
        previous = (ordered[index - 1][3].get("input") or []) if index else None
        if index + 1 < n_calls:
            next_raw_id, _, _, next_row = ordered[index + 1]
            next_items = next_row.get("input") or []
            delta_start = _common_prefix_length(items, next_items)
            output_stop = delta_start
            while output_stop < len(next_items) and is_model_item(next_items[output_stop]):
                output_stop += 1
            output = next_items[delta_start:output_stop]
            delta_location = (next_raw_id, delta_start, len(next_items))
            output_location = (next_raw_id, delta_start, output_stop)
        else:
            output = None
            delta_location = output_location = None
        yield _base_call(
            trajectory_id, index, n_calls, row, raw_id, source_path, source_line,
            items, previous, output, delta_location, output_location, pricing, tier_ranks,
        )


def _history_calls(
    record: Tuple[int, str, int, Dict[str, Any]],
    pricing: Dict[str, Sequence[float]], tier_ranks: Dict[str, int],
) -> Iterator[Dict[str, Any]]:
    raw_id, source_path, source_line, row = record
    items = row.get("input") or []
    spans = decompose(items)
    complete_history = bool(items and spans and is_model_item(items[-1]))
    if not complete_history:
        yield _base_call(
            group_key(row), 0, 1, row, raw_id, source_path, source_line,
            items, None, None, None, None, pricing, tier_ranks,
        )
        return
    trajectory_id = _full_trajectory_id(items)
    for index, (start, stop) in enumerate(spans):
        input_items = items[:start]
        previous_input = items[:spans[index - 1][0]] if index else None
        is_final = index == len(spans) - 1
        output = None if is_final else items[start:stop]
        if is_final:
            delta_location = output_location = None
        else:
            next_start = spans[index + 1][0]
            delta_location = (raw_id, start, next_start)
            output_location = (raw_id, start, stop)
        yield _base_call(
            trajectory_id, index, len(spans), row, raw_id, source_path, source_line,
            input_items, previous_input, output, delta_location, output_location,
            pricing, tier_ranks,
        )


def _build_export_calls(con: Any, pricing: Dict[str, Sequence[float]], progress: Progress) -> None:
    tier_ranks = _tier_ranks(pricing)
    cursor = con.cursor()
    cursor.execute("""SELECT e.group_signature, r.raw_id, r.source_path,
                                    r.source_line, r.raw_json
        FROM export_rows e JOIN raw_lines r USING(raw_id)
        ORDER BY e.group_signature, e.n_items, r.raw_id""")
    current_signature = None
    group: List[Tuple[int, str, int, Dict[str, Any]]] = []
    calls: List[Dict[str, Any]] = []
    groups_done = 0

    def finish_group() -> None:
        nonlocal groups_done
        if not group:
            return
        iterator = (
            _snapshot_calls(group, pricing, tier_ranks)
            if len(group) > 1 else _history_calls(group[0], pricing, tier_ranks)
        )
        calls.extend(iterator)
        groups_done += 1
        if len(calls) >= 2_000:
            _insert_call_batch(con, calls)
        if groups_done % 10_000 == 0:
            _emit(progress, "  derived {:,d} trajectories".format(groups_done))

    while True:
        chunk = cursor.fetchmany(512)
        if not chunk:
            break
        for signature, raw_id, source_path, source_line, raw_json in chunk:
            if current_signature is not None and signature != current_signature:
                finish_group()
                group = []
            current_signature = signature
            group.append((raw_id, source_path, source_line, json.loads(raw_json)))
    finish_group()
    cursor.close()
    _insert_call_batch(con, calls)
    _emit(progress, "Derived {:,d} export trajectories".format(groups_done))


def _consolidate_results(con: Any, pricing: Dict[str, Sequence[float]]) -> None:
    con.execute("""CREATE TABLE ledger AS
        SELECT trajectory_id, call_index,
          arg_max(chosen_model, priority) FILTER (WHERE chosen_model IS NOT NULL) AS chosen_model,
          arg_max(kept_logged, priority) FILTER (WHERE kept_logged IS NOT NULL) AS kept_logged,
          arg_max(direction, priority) FILTER (WHERE direction IS NOT NULL) AS direction,
          arg_max(p_routine, priority) FILTER (WHERE p_routine IS NOT NULL) AS p_routine,
          arg_max(flip_lambda, priority) FILTER (WHERE flip_lambda IS NOT NULL) AS flip_lambda,
          arg_max(cost_usd, priority) FILTER (WHERE cost_usd IS NOT NULL) AS cost_usd,
          arg_max(switch_penalty_usd, priority) FILTER (WHERE switch_penalty_usd IS NOT NULL) AS switch_penalty_usd,
          arg_max(split, priority) FILTER (WHERE split IS NOT NULL) AS split,
          arg_max(routine, priority) FILTER (WHERE routine IS NOT NULL) AS routine,
          arg_max(logged_model, priority) FILTER (WHERE logged_model IS NOT NULL) AS logged_model,
          arg_max(n_calls, priority) FILTER (WHERE n_calls IS NOT NULL) AS n_calls,
          max(ledger_raw_id) AS ledger_raw_id
        FROM ledger_entries GROUP BY trajectory_id, call_index""")
    rows = con.execute("""SELECT r.trajectory_id, r.call_index, r.n_calls,
             r.model, r.model_family, r.est_input_tokens, r.est_new_tokens,
             r.shared_prefix_tokens, r.output_kind, r.output_n_tool_calls,
             r.output_tool_names, r.output_prose_tokens, r.routine, r.raw_id,
             x.source_path, x.source_line
        FROM result_calls r JOIN raw_lines x ON x.raw_id = r.raw_id
        LEFT JOIN calls c ON c.trajectory_id = r.trajectory_id AND c.call_index = r.call_index
        WHERE c.call_id IS NULL
        QUALIFY row_number() OVER (
          PARTITION BY r.trajectory_id, r.call_index ORDER BY r.raw_id DESC) = 1""").fetchall()
    skeletons: List[Dict[str, Any]] = []
    tier_ranks = _tier_ranks(pricing)
    for row in rows:
        (trajectory_id, call_index, n_calls, model, family, inp, new, shared,
         output_kind, output_count, output_names, output_prose, routine, raw_id,
         source_path, source_line) = row
        n_calls = n_calls or call_index + 1
        is_final = call_index == n_calls - 1
        value = {column: None for column in CALL_COLUMNS}
        value.update({
            "call_id": "%s:%d" % (trajectory_id, call_index),
            "trajectory_id": trajectory_id, "call_index": call_index,
            "n_calls_in_trajectory": n_calls, "is_final": is_final,
            "model": model, "model_family": family or _model_family(model),
            "tier_rank": tier_ranks.get(model), "est_input_tokens": inp,
            "est_new_tokens": new, "shared_prefix_tokens": shared,
            "prefix_share": shared / inp if shared is not None and inp else None,
            "output_kind": "unobservable" if is_final else output_kind,
            "output_n_tool_calls": None if is_final else output_count,
            "output_tool_names": None if is_final else output_names,
            "output_prose_tokens": None if is_final else output_prose,
            "routine": None if is_final else routine, "source_kind": "results",
            "source_path": source_path, "source_line": source_line, "raw_id": raw_id,
        })
        skeletons.append(value)
        if len(skeletons) >= 2_000:
            _insert_call_batch(con, skeletons)
    _insert_call_batch(con, skeletons)
    # Route summaries without source calls are useful in results-only mode. If
    # exports are present, unmatched IDs indicate an incompatible/stale ledger;
    # keep them in `ledger` for diagnostics but do not duplicate trajectories.
    has_exports = con.execute("SELECT count(*) > 0 FROM export_rows").fetchone()[0]
    missing = [] if has_exports else con.execute("""SELECT l.trajectory_id,
             l.call_index, l.n_calls, l.logged_model, l.chosen_model,
             l.ledger_raw_id, r.source_path, r.source_line
        FROM ledger l JOIN raw_lines r ON r.raw_id = l.ledger_raw_id
        LEFT JOIN calls c ON c.trajectory_id = l.trajectory_id AND c.call_index = l.call_index
        WHERE c.call_id IS NULL""").fetchall()
    for trajectory_id, call_index, n_calls, model, chosen, raw_id, source_path, source_line in missing:
        n_calls = n_calls or call_index + 1
        value = {column: None for column in CALL_COLUMNS}
        value.update({
            "call_id": "%s:%d" % (trajectory_id, call_index),
            "trajectory_id": trajectory_id, "call_index": call_index,
            "n_calls_in_trajectory": n_calls,
            "is_final": call_index == n_calls - 1, "model": model,
            "model_family": _model_family(model), "tier_rank": tier_ranks.get(model),
            "output_kind": "unobservable" if call_index == n_calls - 1 else None,
            "chosen_model": chosen, "source_kind": "results",
            "source_path": source_path, "source_line": source_line,
            "raw_id": raw_id, "ledger_raw_id": raw_id,
        })
        skeletons.append(value)
        if len(skeletons) >= 2_000:
            _insert_call_batch(con, skeletons)
    _insert_call_batch(con, skeletons)
    con.execute("""UPDATE calls AS c SET
          chosen_model = coalesce(l.chosen_model, c.chosen_model),
          kept_logged = coalesce(l.kept_logged, c.kept_logged),
          direction = coalesce(l.direction, c.direction),
          p_routine = coalesce(l.p_routine, c.p_routine),
          flip_lambda = coalesce(l.flip_lambda, c.flip_lambda),
          cost_usd = coalesce(l.cost_usd, c.cost_usd),
          switch_penalty_usd = coalesce(l.switch_penalty_usd, c.switch_penalty_usd),
          split = coalesce(l.split, c.split),
          routine = CASE WHEN c.is_final THEN NULL ELSE coalesce(l.routine, c.routine) END,
          ledger_raw_id = coalesce(l.ledger_raw_id, c.ledger_raw_id)
        FROM ledger l
        WHERE c.trajectory_id = l.trajectory_id AND c.call_index = l.call_index""")
    models = con.execute("""SELECT DISTINCT model FROM (
        SELECT model FROM calls UNION SELECT chosen_model AS model FROM calls)
        WHERE model IS NOT NULL""").fetchall()
    dense_prices = sorted({price_of(model[0], pricing)[0] for model in models})
    rank_by_price = {price: index + 1 for index, price in enumerate(dense_prices)}
    prices = []
    for (model,) in models:
        pu, pc, po = price_of(model, pricing)
        prices.append((model, pu, pc, po, rank_by_price[pu]))
    con.execute("CREATE TABLE model_prices(model VARCHAR PRIMARY KEY, uncached DOUBLE, "
                "cached DOUBLE, output DOUBLE, tier_rank INTEGER)")
    if prices:
        con.executemany("INSERT INTO model_prices VALUES (?, ?, ?, ?, ?)", prices)
    con.execute("""UPDATE calls AS c SET tier_rank = p.tier_rank
        FROM model_prices p WHERE c.model = p.model""")
    con.execute("""UPDATE calls AS c SET
          kept_logged = coalesce(c.kept_logged, c.chosen_model = c.model),
          direction = coalesce(c.direction,
            CASE WHEN cp.uncached < lp.uncached THEN 'downgraded'
                 WHEN cp.uncached > lp.uncached THEN 'upgraded' ELSE 'kept' END)
        FROM model_prices cp, model_prices lp
        WHERE c.chosen_model IS NOT NULL
          AND cp.model = c.chosen_model AND lp.model = c.model""")
    con.execute("""CREATE TEMP TABLE route_costs AS
        SELECT c.call_id, c.chosen_model IS NOT NULL AS routed,
          coalesce(c.chosen_model, c.model) AS selected_model,
          lag(coalesce(c.chosen_model, c.model)) OVER (
            PARTITION BY c.trajectory_id ORDER BY c.call_index) AS previous_model,
          c.est_input_tokens + coalesce(c.tools_tokens, 0) AS billed_input,
          coalesce(c.shared_prefix_tokens, 0) + coalesce(c.tools_tokens, 0) AS cacheable,
          c.est_output_tokens, p.uncached, p.cached, p.output
        FROM calls c LEFT JOIN model_prices p
          ON p.model = coalesce(c.chosen_model, c.model)""")
    con.execute("""UPDATE calls AS c SET
          switch_penalty_usd = coalesce(c.switch_penalty_usd,
            CASE WHEN r.routed AND r.previous_model IS NOT NULL
                       AND r.previous_model <> r.selected_model
                 THEN r.cacheable * (r.uncached - r.cached) / 1000000.0
                 WHEN r.routed THEN 0.0 ELSE NULL END),
          cost_usd = coalesce(c.cost_usd,
            CASE WHEN r.routed AND r.billed_input IS NOT NULL THEN
              ((r.billed_input - CASE WHEN r.previous_model = r.selected_model
                 THEN least(r.cacheable, r.billed_input) ELSE 0 END) * r.uncached
               + CASE WHEN r.previous_model = r.selected_model
                 THEN least(r.cacheable, r.billed_input) ELSE 0 END * r.cached
               + coalesce(r.est_output_tokens, 0) * r.output) / 1000000.0
            ELSE NULL END)
        FROM route_costs r WHERE c.call_id = r.call_id""")


def _build_trajectories(con: Any) -> None:
    con.execute("""CREATE TABLE trajectories AS
        SELECT trajectory_id, arg_min(model, call_index) AS model,
          arg_min(model_family, call_index) AS model_family,
          max(n_calls_in_trajectory) AS n_calls,
          sum(est_input_tokens) AS total_est_input_tokens,
          sum(est_new_tokens) AS total_est_new_tokens,
          sum(shared_prefix_tokens) AS total_shared_prefix_tokens,
          sum(est_output_tokens) AS total_est_output_tokens,
          arg_min(tools_tokens, call_index) AS tools_tokens,
          arg_min(tools_available, call_index) AS tools_available,
          list(coalesce(output_kind, 'unobservable') ORDER BY call_index) AS output_kinds,
          flatten(list(coalesce(output_tool_names, []::VARCHAR[]) ORDER BY call_index)) AS tool_calls,
          count(*) FILTER (WHERE tool_output_error) AS n_error_calls,
          string_agg(item_type_sequence, '|' ORDER BY call_index)
            FILTER (WHERE item_type_sequence IS NOT NULL) AS item_type_sequence,
          bool_or(tool_output_error) AS has_error,
          bool_or(repeat_tool) AS has_repeat_tool,
          sum(logged_input_cost_usd) AS logged_cost_usd,
          sum(logged_output_cost_usd) AS logged_output_cost_usd,
          sum(logged_total_cost_usd) AS logged_total_cost_usd,
          arg_min(logged_total_cost_usd, call_index) AS cold_start_cost_usd,
          CASE WHEN count(cost_usd) > 0 THEN sum(cost_usd) ELSE NULL END AS routed_cost_usd,
          arg_min(o.opening_user, call_index) AS opening_user,
          arg_min(o.system_prompt, call_index) AS system_prompt,
          arg_min(source_kind, call_index) AS source_kind,
          arg_min(source_path, call_index) AS source_path,
          arg_min(source_line, call_index) AS source_line
        FROM calls c LEFT JOIN raw_openings o ON o.raw_id = c.raw_id
        GROUP BY c.trajectory_id""")
    if not con.execute("SELECT count(*) > 0 FROM export_rows").fetchone()[0]:
        con.execute("""INSERT INTO trajectories
            SELECT r.trajectory_id, arg_max(r.model, r.raw_id),
              arg_max(r.model_family, r.raw_id), arg_max(r.n_calls, r.raw_id),
              NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
              NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
              'results', arg_max(x.source_path, r.raw_id), arg_max(x.source_line, r.raw_id)
            FROM result_trajectories r JOIN raw_lines x ON x.raw_id = r.raw_id
            LEFT JOIN trajectories t USING(trajectory_id)
            WHERE t.trajectory_id IS NULL GROUP BY r.trajectory_id""")
    con.execute("CREATE INDEX calls_trajectory_idx ON calls(trajectory_id)")
    con.execute("CREATE INDEX calls_model_idx ON calls(model)")
    con.execute("CREATE INDEX calls_output_kind_idx ON calls(output_kind)")
    con.execute("CREATE INDEX trajectories_id_idx ON trajectories(trajectory_id)")
    con.execute("ANALYZE")


# --- Trajectory classification -------------------------------------------------
# The export ships no labels, but the opening user message is not anonymous: Viktor
# packs a `# === Thread info ===` block and a cron memory header into it. Between
# them they yield the trigger, the cron identity and a wall-clock timestamp. See
# scripts/trajectory_ids.py for the verification and the redaction caveat below.
PATH_RE = re.compile(r"^\s*-\s*Path:\s*`([^`]+)`", re.M)
TRIGGER_RE = re.compile(r"^\s*-\s*Triggered by:\s*`([^`]+)`", re.M)
DATETIME_RE = re.compile(r"^\s*-\s*Current datetime:\s*`([^`]+)`", re.M)
CRON_PATH_RE = re.compile(r"^\s*-\s*Cron path:\s*`([^`]+)`", re.M)
PII_RE = re.compile(r"PII_[A-Z_]+_\d+")

# Tool calls grouped by what the model was actually doing with the turn.
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
    "submit_subagent_result": "message",
    "wait_for_background_work": "wait",
}
SIZE_BANDS = ((10_000, "XS"), (50_000, "S"), (200_000, "M"), (1_000_000, "L"))
DEPTH_BANDS = ((1, "single"), (4, "short"), (9, "medium"), (24, "long"))
OUTPUT_KIND_GLYPH = {
    "tool_call": "T", "multi_tool": "M", "patch": "P",
    "text": "S", "unobservable": "X", "unknown": "?",
}


def _band(value: Optional[int], bands: Sequence[Tuple[int, str]], top: str) -> str:
    if value is None:
        return "unknown"
    for limit, name in bands:
        if value <= limit:
            return name
    return top


def _cron_family(cron_path: Optional[str]) -> Tuple[Optional[str], bool]:
    """Leading segment of the cron path, plus whether redaction destroyed it.

    Placeholders are only consistent *within* a trajectory, so a family that is
    itself a `PII_*` token is not a real group — it silently merges unrelated
    crons. Flag it rather than trust it.
    """
    if not cron_path:
        return None, False
    parts = [part for part in cron_path.strip("/").split("/") if part]
    if parts and parts[0] == "crons":
        parts = parts[1:]
    if not parts:
        return None, False
    family = parts[0]
    return family, bool(PII_RE.fullmatch(family))


def _shape_signature(output_kinds: Sequence[str]) -> str:
    """Run-length-compressed turn shape, e.g. `3T·M·2T·X`. Two trajectories with
    the same signature took structurally the same path through the agent loop."""
    parts, run, previous = [], 0, None
    for kind in list(output_kinds) + [None]:
        glyph = OUTPUT_KIND_GLYPH.get(kind or "", "?") if kind is not None else None
        if glyph == previous:
            run += 1
            continue
        if previous is not None:
            parts.append(previous if run == 1 else "%d%s" % (run, previous))
        previous, run = glyph, 1
    return "·".join(parts)


def _classify_trajectories(con: Any) -> None:
    con.execute("""CREATE TABLE trajectory_meta(
        trajectory_id VARCHAR PRIMARY KEY, trigger_kind VARCHAR,
        cron_path VARCHAR, cron_family VARCHAR, cron_family_collapsed BOOLEAN,
        thread_path VARCHAR, started_at VARCHAR, harness VARCHAR, surface VARCHAR,
        toolset_id VARCHAR, toolset_label VARCHAR, workload VARCHAR,
        size_band VARCHAR, depth_band VARCHAR, shape_signature VARCHAR,
        tool_signature VARCHAR, n_tool_calls INTEGER, work_mix VARCHAR,
        billed_input_tokens BIGINT, cache_hit_rate DOUBLE)""")
    rows = con.execute("""SELECT trajectory_id, opening_user, tools_available,
          output_kinds, tool_calls, n_calls, total_est_input_tokens,
          tools_tokens, total_shared_prefix_tokens
        FROM trajectories""").fetchall()
    toolset_labels: Dict[str, str] = {}
    records = []
    for (traj, opening, tools, kinds, tool_calls, n_calls, input_tokens,
         tools_tokens, shared) in rows:
        opening = opening or ""
        tools = list(tools or [])
        tool_calls = [name for name in (tool_calls or []) if name]
        cron_path = _first_group(CRON_PATH_RE, opening)
        family, collapsed = _cron_family(cron_path)
        trigger = _first_group(TRIGGER_RE, opening)
        if trigger in (None, ""):
            trigger_kind = "unknown"
        elif trigger in ("cron", "trigger"):
            trigger_kind = trigger
        else:
            trigger_kind = "user"
        harness = "shell+patch" if "shell_command" in tools else (
            "bash+file" if "bash" in tools else "other")
        surface = "msteams" if any(t.startswith("coworker_") and "msteams" in t for t in tools) else (
            "slack" if any(t.startswith("coworker_") and "slack" in t for t in tools) else "none")
        toolset_id = hashlib.sha1(",".join(sorted(tools)).encode()).hexdigest()[:8]
        toolset_labels.setdefault(toolset_id, "%s / %s / %d tools" % (harness, surface, len(tools)))
        work = [TOOL_WORK.get(name, "other") for name in tool_calls]
        counts: Dict[str, int] = {}
        for kind in work:
            counts[kind] = counts.get(kind, 0) + 1
        workload = max(counts, key=lambda k: (counts[k], k)) if counts else "none"
        billed_input = (input_tokens or 0) + (tools_tokens or 0) * (n_calls or 1)
        cacheable = (shared or 0) + (tools_tokens or 0) * max((n_calls or 1) - 1, 0)
        records.append((
            traj, trigger_kind, cron_path, family, collapsed,
            _first_group(PATH_RE, opening), _first_group(DATETIME_RE, opening),
            harness, surface, toolset_id, toolset_labels[toolset_id], workload,
            _band(billed_input, SIZE_BANDS, "XL"), _band(n_calls, DEPTH_BANDS, "deep"),
            _shape_signature(kinds or []),
            ",".join(_ordered_unique(tool_calls)), len(tool_calls),
            ",".join("%s:%d" % (k, counts[k]) for k in sorted(counts, key=lambda k: -counts[k])),
            billed_input, (cacheable / billed_input) if billed_input else None,
        ))
    if records:
        con.executemany(
            "INSERT INTO trajectory_meta VALUES (%s)" % ", ".join("?" * 20), records,
        )
    con.execute("CREATE INDEX trajectory_meta_idx ON trajectory_meta(trajectory_id)")


def _first_group(pattern: "re.Pattern[str]", text: str) -> Optional[str]:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


# Argument keys that carry the command for the shell-ish tools. The leading verb of
# that command is the closest thing the export has to "what was this call for".
COMMAND_KEYS = ("command", "cmd", "script", "shell_command")
# Wrappers that prefix a real command; skip past them to find the verb that matters.
# `cd /work && cat x` must report `cat`, not `work`: these consume a path argument.
PATH_TAKING = {"cd", "source", ".", "pushd"}
COMMAND_NOISE = {
    "sudo", "env", "time", "nohup", "exec", "then", "do", "else",
    "&&", "||", ";", "(", "{", "\\",
}


def _command_verb(argument_text: Optional[str]) -> Optional[str]:
    """Leading executable of a shell command, looking past `cd x &&` style prefixes.

    A heuristic, and it says so: a pipeline is reduced to its first stage, and a
    heredoc to the command that opens it. Good enough to cluster 5k bash calls into
    something a human can read; not a parser.
    """
    if not argument_text:
        return None
    try:
        payload = json.loads(argument_text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    command = None
    if isinstance(payload, dict):
        for key in COMMAND_KEYS:
            if isinstance(payload.get(key), str):
                command = payload[key]
                break
    elif isinstance(payload, str):
        command = payload
    if command is None:
        return None
    skip_next = False
    for token in re.split(r"[\s|;]+", command.strip()):
        token = token.strip("()&")
        if not token:
            continue
        if skip_next:            # the path argument of cd/source, not a command
            skip_next = False
            continue
        if token in PATH_TAKING:
            skip_next = True
            continue
        if token in COMMAND_NOISE or "=" in token or token.startswith("-"):
            continue
        return token.rsplit("/", 1)[-1][:40]
    return None


def _argument_shape(argument_text: Optional[str]) -> Optional[str]:
    """Sorted top-level argument keys — the call's signature, ignoring its values."""
    if not argument_text:
        return None
    try:
        payload = json.loads(argument_text)
    except (json.JSONDecodeError, TypeError):
        return "(unparsed)"
    if isinstance(payload, dict):
        return ",".join(sorted(payload)) or "(empty)"
    return "(%s)" % type(payload).__name__


def _build_function_calls(con: Any, progress: Progress) -> None:
    """One row per function call a model actually emitted, with its result and cost.

    Every tool call sits inside some request's recovered output span, so it can be
    attributed to the model that produced it and to that request's price. A request
    that emits several calls at once has one cost; `cost_share_usd` splits it evenly
    across them, which is an attribution choice, not a measurement.
    """
    con.execute("""CREATE TABLE function_calls AS
        WITH ranked AS (
          SELECT raw_id, item_index, item_tokens, search_field, search_text, tool_name,
                 is_model_authored AND tool_name IS NOT NULL AS is_call,
                 count(*) FILTER (WHERE is_model_authored AND tool_name IS NOT NULL)
                   OVER (PARTITION BY raw_id ORDER BY item_index
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS call_rank
          FROM raw_items
        ),
        calls_only AS (
          SELECT raw_id, item_index, call_rank, tool_name AS tool,
                 item_tokens AS arg_tokens, search_text AS arg_text
          FROM ranked WHERE is_call
        ),
        results_only AS (
          SELECT raw_id, call_rank,
                 arg_min(item_tokens, item_index) AS result_tokens,
                 arg_min(search_text, item_index) AS result_text
          FROM ranked
          WHERE NOT is_call AND search_field = 'tool_outputs' AND call_rank > 0
          GROUP BY raw_id, call_rank
        )
        SELECT
          c.raw_id || ':' || c.item_index AS fc_id,
          o.call_id, o.trajectory_id, o.call_index, o.model, o.model_family,
          o.n_calls_in_trajectory, c.tool, c.arg_tokens, c.arg_text,
          r.result_tokens, r.result_text,
          coalesce(o.output_n_tool_calls, 1) AS n_calls_in_request,
          o.logged_total_cost_usd AS request_cost_usd,
          o.logged_total_cost_usd / greatest(coalesce(o.output_n_tool_calls, 1), 1)
            AS cost_share_usd,
          o.est_input_tokens + coalesce(o.tools_tokens, 0) AS request_input_tokens,
          o.est_output_tokens AS request_output_tokens,
          c.raw_id, c.item_index
        FROM calls_only c
        JOIN calls o
          ON o.output_raw_id = c.raw_id
         AND c.item_index >= o.output_start_item
         AND c.item_index <  o.output_stop_item
        LEFT JOIN results_only r
          ON r.raw_id = c.raw_id AND r.call_rank = c.call_rank""")
    con.execute("ALTER TABLE function_calls ADD COLUMN command_verb VARCHAR")
    con.execute("ALTER TABLE function_calls ADD COLUMN arg_shape VARCHAR")
    con.execute("ALTER TABLE function_calls ADD COLUMN result_error BOOLEAN")
    rows = con.execute(
        "SELECT fc_id, arg_text, result_text FROM function_calls").fetchall()
    updates = [
        (
            _command_verb(arg_text), _argument_shape(arg_text),
            bool(result_text and ERROR_RE.search(result_text)), fc_id,
        )
        for fc_id, arg_text, result_text in rows
    ]
    if updates:
        con.executemany(
            "UPDATE function_calls SET command_verb = ?, arg_shape = ?, "
            "result_error = ? WHERE fc_id = ?", updates,
        )
    con.execute("CREATE INDEX function_calls_tool_idx ON function_calls(tool)")
    con.execute("CREATE INDEX function_calls_model_idx ON function_calls(model)")
    _emit(progress, "Grouped {:,d} function calls".format(len(rows)))


def _rebuild(
    root: Path, db_path: Path, sources: Sequence[SourceFile],
    progress: Progress, started: float,
) -> IndexResult:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required; install requirements.txt first") from exc
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_name(".%s.%d.tmp" % (db_path.name, os.getpid()))
    if temp_path.exists():
        temp_path.unlink()
    _emit(progress, "Index manifest changed; rebuilding %s" % db_path)
    con = duckdb.connect(str(temp_path))
    try:
        _create_schema(con)
        con.execute("INSERT INTO index_meta VALUES ('index_version', ?)", [INDEX_VERSION])
        con.execute("INSERT INTO index_meta VALUES ('token_estimator', 'serialized JSON chars / 4')")
        _ingest_sources(con, sources, progress)
        pricing = load_pricing()
        _build_export_calls(con, pricing, progress)
        _consolidate_results(con, pricing)
        _build_trajectories(con)
        _classify_trajectories(con)
        _build_function_calls(con, progress)
        calls = con.execute("SELECT count(*) FROM calls").fetchone()[0]
        trajectories = con.execute("SELECT count(*) FROM trajectories").fetchone()[0]
        con.execute("CHECKPOINT")
        con.close()
        os.replace(str(temp_path), str(db_path))
    except Exception:
        con.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
    elapsed = time.monotonic() - started
    _emit(progress, "Indexed {:,d} calls in {:,d} trajectories ({:.2f}s)".format(
        calls, trajectories, elapsed,
    ))
    return IndexResult(
        True, calls, trajectories,
        sum(source.kind == "export" for source in sources),
        sum(source.kind == "results" for source in sources),
        elapsed, "source manifest changed",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    result = ensure_index(force="--force" in argv)
    status = "rebuilt" if result.rebuilt else "current"
    print("index {}: {:,d} calls, {:,d} trajectories; export files={}, result files={}".format(
        status, result.calls, result.trajectories,
        result.export_files, result.result_files,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
