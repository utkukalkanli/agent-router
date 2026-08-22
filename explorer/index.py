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
INDEX_VERSION = "4"
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
    "ledger_raw_id", "logged_input_cost_usd",
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
        logged_input_cost_usd DOUBLE)""")


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
    if item_type in ("function_call", "custom_tool_call") and "arguments" in item:
        search_field = "tool_arguments"
        value = item.get("arguments")
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
    pu, pc, _ = price_of(model, pricing)
    logged_cost = (
        max(input_tokens - shared, 0) * pu + min(shared, input_tokens) * pc
    ) / 1e6
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
        pu, pc, _ = price_of(model, pricing)
        prices.append((model, pu, pc, rank_by_price[pu]))
    con.execute("CREATE TABLE model_prices(model VARCHAR PRIMARY KEY, uncached DOUBLE, cached DOUBLE, tier_rank INTEGER)")
    if prices:
        con.executemany("INSERT INTO model_prices VALUES (?, ?, ?, ?)", prices)
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
          c.est_input_tokens, c.shared_prefix_tokens, p.uncached, p.cached
        FROM calls c LEFT JOIN model_prices p
          ON p.model = coalesce(c.chosen_model, c.model)""")
    con.execute("""UPDATE calls AS c SET
          switch_penalty_usd = coalesce(c.switch_penalty_usd,
            CASE WHEN r.routed AND r.previous_model IS NOT NULL
                       AND r.previous_model <> r.selected_model
                 THEN coalesce(r.shared_prefix_tokens, 0) *
                      (r.uncached - r.cached) / 1000000.0
                 WHEN r.routed THEN 0.0 ELSE NULL END),
          cost_usd = coalesce(c.cost_usd,
            CASE WHEN r.routed AND r.est_input_tokens IS NOT NULL THEN
              ((r.est_input_tokens - CASE WHEN r.previous_model = r.selected_model
                 THEN least(coalesce(r.shared_prefix_tokens, 0), r.est_input_tokens)
                 ELSE 0 END) * r.uncached
               + CASE WHEN r.previous_model = r.selected_model
                 THEN least(coalesce(r.shared_prefix_tokens, 0), r.est_input_tokens)
                 ELSE 0 END * r.cached) / 1000000.0 ELSE NULL END)
        FROM route_costs r WHERE c.call_id = r.call_id""")


def _build_trajectories(con: Any) -> None:
    con.execute("""CREATE TABLE trajectories AS
        SELECT trajectory_id, arg_min(model, call_index) AS model,
          arg_min(model_family, call_index) AS model_family,
          max(n_calls_in_trajectory) AS n_calls,
          sum(est_input_tokens) AS total_est_input_tokens,
          sum(est_new_tokens) AS total_est_new_tokens,
          sum(shared_prefix_tokens) AS total_shared_prefix_tokens,
          string_agg(item_type_sequence, '|' ORDER BY call_index)
            FILTER (WHERE item_type_sequence IS NOT NULL) AS item_type_sequence,
          bool_or(tool_output_error) AS has_error,
          bool_or(repeat_tool) AS has_repeat_tool,
          sum(logged_input_cost_usd) AS logged_cost_usd,
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
              'results', arg_max(x.source_path, r.raw_id), arg_max(x.source_line, r.raw_id)
            FROM result_trajectories r JOIN raw_lines x ON x.raw_id = r.raw_id
            LEFT JOIN trajectories t USING(trajectory_id)
            WHERE t.trajectory_id IS NULL GROUP BY r.trajectory_id""")
    con.execute("CREATE INDEX calls_trajectory_idx ON calls(trajectory_id)")
    con.execute("CREATE INDEX calls_model_idx ON calls(model)")
    con.execute("CREATE INDEX calls_output_kind_idx ON calls(output_kind)")
    con.execute("CREATE INDEX trajectories_id_idx ON trajectories(trajectory_id)")
    con.execute("ANALYZE")


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
