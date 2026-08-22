import json
import os
import tempfile
import unittest
from pathlib import Path

import duckdb

from explorer.index import ensure_index
from scripts.load_trajectories import group_key
from scripts.trajectory_ids import sha


def message(role, text):
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def request(model, items):
    return {
        "model": model,
        "input": items,
        "tools": [{
            "type": "function", "name": "shell",
            "parameters": {"type": "object", "properties": {}},
        }],
    }


class ExplorerIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "explorer" / "index.duckdb"

    def tearDown(self):
        self.temp.cleanup()

    def write_jsonl(self, relative, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = [json.dumps(row, separators=(",", ":")) for row in rows]
        path.write_text("\n".join(encoded) + "\n", encoding="utf-8")
        return path, encoded

    def fixture_exports(self):
        base = [message("system", "snapshot system"), message("user", "snapshot task")]
        first = request("gpt-5.6-terra", base)
        second = request("gpt-5.6-terra", base + [
            {"type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"ok\"}"},
            {"type": "function_call_output", "output": "ok"},
        ])
        history_items = [
            message("system", "history system"), message("user", "history task"),
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"bad\"}"},
            {"type": "function_call_output", "output": "ERROR: failed"},
            message("assistant", "Recovered."),
        ]
        history = request("claude-sonnet-5", history_items)
        return first, second, history

    def test_ids_derived_columns_final_nulls_and_raw_provenance(self):
        first, second, history = self.fixture_exports()
        _, encoded = self.write_jsonl("export/sample.jsonl", [first, second, history])
        result = ensure_index(self.root, self.db, progress=None)
        self.assertEqual((result.calls, result.trajectories), (4, 2))

        con = duckdb.connect(str(self.db), read_only=True)
        ids = {row[0] for row in con.execute("SELECT trajectory_id FROM trajectories").fetchall()}
        # No trace.py exists in this checkout. These are its two authoritative
        # predecessors: load_trajectories for snapshots and trajectory_ids /
        # decompose for complete-history rows.
        self.assertEqual(ids, {group_key(first), sha(history["input"], 12)})

        final_bad = con.execute("""SELECT count(*) FROM calls WHERE is_final AND (
            output_kind <> 'unobservable' OR output_n_tool_calls IS NOT NULL OR
            output_tool_names IS NOT NULL OR output_prose_tokens IS NOT NULL OR
            routine IS NOT NULL)""").fetchone()[0]
        self.assertEqual(final_bad, 0)
        first_call = con.execute("""SELECT output_kind, output_n_tool_calls,
            output_tool_names, item_type_sequence, last_item_type
            FROM calls WHERE trajectory_id = ? AND call_index = 0""", [group_key(first)]).fetchone()
        self.assertEqual(first_call, ("tool_call", 1, ["shell"], "sys,user", "user"))
        self.assertEqual(
            con.execute("SELECT raw_json FROM raw_lines WHERE source_line = 1").fetchone()[0],
            encoded[0],
        )
        con.close()

    def test_ledger_join_and_switch_penalty(self):
        first, second, _ = self.fixture_exports()
        self.write_jsonl("export/sample.jsonl", [first, second])
        trajectory_id = group_key(first)
        self.write_jsonl("results/router.jsonl", [{
            "trajectory_id": trajectory_id,
            "n_calls": 2,
            "logged_model": "gpt-5.6-terra",
            "route": ["gpt-5.6-luna", "gpt-5.6-sol"],
            "p_routine": 0.8,
            "flip_lambda": 0.2,
            "split": "test",
            "routine": True,
        }])
        ensure_index(self.root, self.db, progress=None)
        con = duckdb.connect(str(self.db), read_only=True)
        rows = con.execute("""SELECT chosen_model, kept_logged, direction,
            p_routine, flip_lambda, split, routine, switch_penalty_usd
            FROM calls ORDER BY call_index""").fetchall()
        self.assertEqual(rows[0][:7], (
            "gpt-5.6-luna", False, "downgraded", 0.8, 0.2, "test", True,
        ))
        self.assertEqual(rows[1][0:3], ("gpt-5.6-sol", False, "upgraded"))
        self.assertIsNone(rows[1][6])
        self.assertGreater(rows[1][7], 0)
        con.close()

    def test_export_only_and_incremental_manifest(self):
        first, second, _ = self.fixture_exports()
        source, _ = self.write_jsonl("export/sample.jsonl", [first, second])
        built = ensure_index(self.root, self.db, progress=None)
        current = ensure_index(self.root, self.db, progress=None)
        self.assertTrue(built.rebuilt)
        self.assertFalse(current.rebuilt)
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        changed = ensure_index(self.root, self.db, progress=None)
        self.assertTrue(changed.rebuilt)
        self.assertEqual((changed.export_files, changed.result_files), (1, 0))

    def test_results_only_builds_skeleton_calls(self):
        self.write_jsonl("results/router.jsonl", [{
            "trajectory": "result-only",
            "n_calls": 2,
            "logged_model": "claude-opus-5",
            "route": ["claude-sonnet-5", "claude-sonnet-5"],
        }])
        result = ensure_index(self.root, self.db, progress=None)
        self.assertEqual((result.calls, result.trajectories), (2, 1))
        self.assertEqual((result.export_files, result.result_files), (0, 1))
        con = duckdb.connect(str(self.db), read_only=True)
        row = con.execute("""SELECT source_kind, output_kind,
            output_n_tool_calls FROM calls ORDER BY call_index DESC LIMIT 1""").fetchone()
        self.assertEqual(row, ("results", "unobservable", None))
        con.close()


if __name__ == "__main__":
    unittest.main()
