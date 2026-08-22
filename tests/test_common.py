import json
from pathlib import Path

from _common import write_routes

SYS = {"role": "system", "content": "You are Viktor."}
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello task"}]}


def make_export(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    req1 = {"model": "claude-opus-5", "input": [SYS, USER], "tools": []}
    req2 = {
        "model": "claude-opus-5",
        "input": [SYS, USER, {"type": "function_call", "name": "x", "call_id": "1"},
                   {"type": "function_call_output", "call_id": "1", "output": "ok"}],
        "tools": [],
    }
    with open(export / "trajectories_v1_01.jsonl", "w") as f:
        f.write(json.dumps(req1) + "\n")
        f.write(json.dumps(req2) + "\n")
    return export


def always_sonnet(calls):
    return ["claude-sonnet-5" for _ in calls]


def test_write_routes_writes_expected_schema(tmp_path):
    export = make_export(tmp_path)
    results_root = tmp_path / "results"

    write_routes(always_sonnet, export=str(export), name="test_version", results_root=results_root)

    out_file = results_root / "test_version" / "routes.jsonl"
    assert out_file.exists()
    lines = [json.loads(l) for l in out_file.read_text().splitlines()]
    assert len(lines) == 1  # one trajectory reconstructed from the two calls

    rec = lines[0]
    assert set(rec.keys()) == {"trajectory", "n_calls", "logged_model", "route",
                                "cost_logged_usd", "cost_routed_usd", "switches"}
    assert rec["n_calls"] == 2
    assert rec["logged_model"] == "claude-opus-5"
    assert rec["route"] == ["claude-sonnet-5", "claude-sonnet-5"]
    assert rec["switches"] == 0
    assert isinstance(rec["cost_logged_usd"], float)
    assert isinstance(rec["cost_routed_usd"], float)


def test_write_routes_derives_name_from_calling_script(tmp_path, monkeypatch):
    export = make_export(tmp_path)
    results_root = tmp_path / "results"
    monkeypatch.setattr("sys.argv", [str(tmp_path / "v1_heuristic.py")])

    write_routes(always_sonnet, export=str(export), results_root=results_root)

    assert (results_root / "v1_heuristic" / "routes.jsonl").exists()
