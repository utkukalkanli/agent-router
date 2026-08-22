"""FastAPI server for the local trajectory explorer."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import analysis
from .index import DB_PATH, ROOT, ensure_index


STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_COLUMNS = [
    "call_id", "trajectory_id", "call_index", "n_calls_in_trajectory",
    "model", "model_family", "tier_rank", "est_input_tokens",
    "est_new_tokens", "shared_prefix_tokens", "prefix_share", "n_items",
    "last_item_type", "last_tool_name", "n_distinct_tools", "has_image",
    "has_reasoning", "tool_output_error", "repeat_tool", "output_kind",
    "output_n_tool_calls", "output_tool_names", "output_prose_tokens",
    "routine", "chosen_model", "kept_logged", "direction", "p_routine",
    "flip_lambda", "cost_usd", "switch_penalty_usd", "split",
]


def _connect(db_path: Path):
    return duckdb.connect(str(db_path), read_only=True)


def _column_info(con: Any, table: str = "calls") -> List[Dict[str, Any]]:
    return [
        {"name": row[1], "type": row[2], "nullable": not row[3]}
        for row in con.execute("PRAGMA table_info('%s')" % table).fetchall()
    ]


def _quote_identifier(value: str) -> str:
    return '"%s"' % value.replace('"', '""')


def _json_rows(cursor: Any) -> List[Dict[str, Any]]:
    names = [description[0] for description in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def create_app(db_path: Path = DB_PATH, static_dir: Path = STATIC_DIR) -> FastAPI:
    db_path, static_dir = Path(db_path), Path(static_dir)
    app = FastAPI(
        title="Viktor Trajectory Explorer",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.db_path = db_path

    @app.get("/api/schema")
    def schema() -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            columns = _column_info(con)
            sources = _json_rows(con.execute(
                """SELECT kind, path, line_count, parse_errors, size_bytes,
                          mtime_ns
                   FROM sources ORDER BY kind, path"""
            ))
            counts = con.execute(
                "SELECT (SELECT count(*) FROM calls), "
                "(SELECT count(*) FROM trajectories)"
            ).fetchone()
            estimator = con.execute(
                "SELECT value FROM index_meta WHERE key = 'token_estimator'"
            ).fetchone()
        return {
            "columns": columns,
            "default_columns": DEFAULT_COLUMNS,
            "sources": sources,
            "source_kinds": sorted({source["kind"] for source in sources}),
            "calls": counts[0],
            "trajectories": counts[1],
            "token_estimator": estimator[0] if estimator else None,
        }

    @app.get("/api/calls")
    def calls(
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=500),
        sort: str = Query("call_id"),
        order: str = Query("asc", pattern="^(asc|desc)$"),
        columns: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        with _connect(app.state.db_path) as con:
            available = {column["name"] for column in _column_info(con)}
            if sort not in available:
                raise HTTPException(400, "unknown sort column: %s" % sort)
            selected = [part.strip() for part in columns.split(",")] if columns else list(DEFAULT_COLUMNS)
            selected = [column for column in selected if column]
            unknown = sorted(set(selected) - available)
            if unknown:
                raise HTTPException(400, "unknown columns: %s" % ", ".join(unknown))
            if "call_id" not in selected:
                selected.insert(0, "call_id")
            selected = list(dict.fromkeys(selected))
            select_sql = ", ".join(_quote_identifier(column) for column in selected)
            sort_sql = _quote_identifier(sort)
            cursor = con.execute(
                "SELECT %s FROM calls ORDER BY %s %s NULLS LAST, call_id ASC "
                "LIMIT ? OFFSET ?" % (select_sql, sort_sql, order.upper()),
                [limit, offset],
            )
            rows = _json_rows(cursor)
            total = con.execute("SELECT count(*) FROM calls").fetchone()[0]
        return {
            "rows": rows,
            "total": total,
            "offset": offset,
            "limit": limit,
            "sort": sort,
            "order": order,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @app.get("/api/overview")
    def overview() -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            return analysis.overview(con)

    @app.get("/api/classes")
    def classes(axis: Optional[str] = Query(None)) -> Dict[str, Any]:
        if axis and axis not in analysis.AXIS_BY_KEY:
            raise HTTPException(400, "unknown axis: %s" % axis)
        with _connect(app.state.db_path) as con:
            return analysis.classes(con, axis)

    @app.get("/api/tools")
    def tools() -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            return analysis.tool_profile(con)

    @app.get("/api/trajectories")
    def trajectories(
        sort: str = Query("cost_usd"),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        q: Optional[str] = Query(None),
        model: Optional[str] = None, workload: Optional[str] = None,
        trigger_kind: Optional[str] = None, harness: Optional[str] = None,
        surface: Optional[str] = None, size_band: Optional[str] = None,
        depth_band: Optional[str] = None, toolset_label: Optional[str] = None,
        cron_family: Optional[str] = None, shape_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        if sort not in analysis.SORTABLE:
            raise HTTPException(400, "unknown sort: %s" % sort)
        filters = {
            "model": model, "workload": workload, "trigger_kind": trigger_kind,
            "harness": harness, "surface": surface, "size_band": size_band,
            "depth_band": depth_band, "toolset_label": toolset_label,
            "cron_family": cron_family, "shape_signature": shape_signature,
        }
        with _connect(app.state.db_path) as con:
            return analysis.trajectory_list(
                con, filters, q, sort, order, limit, offset)

    @app.get("/api/trajectories/{trajectory_id}")
    def trajectory(trajectory_id: str) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            detail = analysis.trajectory_detail(con, trajectory_id)
        if not detail:
            raise HTTPException(404, "unknown trajectory: %s" % trajectory_id)
        return detail

    @app.get("/api/trajectories/{trajectory_id}/similar")
    def similar(trajectory_id: str, limit: int = Query(12, ge=1, le=50)) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            result = analysis.similar(con, trajectory_id, limit)
        if not result:
            raise HTTPException(404, "unknown trajectory: %s" % trajectory_id)
        return result

    @app.get("/api/compare")
    def compare(a: str, b: str) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            result = analysis.compare(con, a, b)
        if not result:
            raise HTTPException(404, "unknown trajectory in %s, %s" % (a, b))
        return result

    @app.get("/api/functions")
    def functions() -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            return analysis.function_groups(con)

    @app.get("/api/functions/{tool}")
    def function_detail(
        tool: str,
        group_by: str = Query("command_verb", pattern="^(command_verb|arg_shape)$"),
        limit: int = Query(40, ge=1, le=200),
    ) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            result = analysis.function_detail(con, tool, group_by, limit)
        if not result:
            raise HTTPException(404, "no calls to tool: %s" % tool)
        return result

    @app.get("/api/functions/{tool}/samples")
    def function_samples(
        tool: str,
        group_key: Optional[str] = Query(None),
        group_by: str = Query("command_verb", pattern="^(command_verb|arg_shape)$"),
        per_model: int = Query(3, ge=1, le=10),
    ) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            return analysis.function_samples(con, tool, group_key, group_by, per_model)

    @app.get("/api/sources")
    def sources_view() -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            return analysis.source_files(con)

    @app.get("/api/sources/rows")
    def source_rows(
        path: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            result = analysis.source_rows(con, path, offset, limit)
        if not result:
            raise HTTPException(404, "unknown source file: %s" % path)
        return result

    @app.get("/api/sources/row")
    def source_row(path: str, line: int = Query(..., ge=1)) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            result = analysis.source_row(con, path, line)
        if not result:
            raise HTTPException(404, "no line %d in %s" % (line, path))
        return result

    @app.get("/api/matched")
    def matched(
        axes: str = Query("toolset_label,workload,size_band"),
        min_models: int = Query(2, ge=2, le=9),
    ) -> Dict[str, Any]:
        with _connect(app.state.db_path) as con:
            return analysis.matched_cohorts(
                con, [part.strip() for part in axes.split(",") if part.strip()],
                min_models)

    if static_dir.is_dir() and (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        def missing_frontend() -> str:
            return (
                "<h1>Trajectory index is ready</h1>"
                "<p>Build explorer/frontend to enable the Calls UI.</p>"
                "<p><a href='/api/docs'>API documentation</a></p>"
            )
    return app


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Index and serve the local trajectory explorer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = ensure_index(ROOT, DB_PATH, force=args.force_index, progress=print)
    status = "rebuilt" if result.rebuilt else "current"
    print("Index {}: {:,} calls in {:,} trajectories".format(
        status, result.calls, result.trajectories,
    ))
    print("Serving locally at http://{}:{} (no telemetry or external requests)".format(
        args.host, args.port,
    ))
    import uvicorn
    uvicorn.run(create_app(DB_PATH), host=args.host, port=args.port, log_level="info")
    return 0
