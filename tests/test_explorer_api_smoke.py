"""Every read endpoint must answer 200 on a real-shaped export.

The analysis queries are string-built, so a typo in a column alias only shows up
when the endpoint is actually called. This walks the whole surface once.
"""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from explorer.index import ensure_index
from explorer.server import STATIC_DIR, create_app


def message(role, text):
    return {"type": "message", "role": role,
            "content": [{"type": "input_text", "text": text}]}


def trajectory(model, commands, task):
    """A complete trajectory: system + user, then bash call / result per command.

    Every row needs its OWN opening user message. The indexer groups rows that share
    one, treating them as snapshots of a single trajectory — which is what the starter
    loader assumed and what the real export never does.
    """
    items = [
        message("system", "you are an agent"),
        message("user", "# Thread info\n- Triggered by: `cron`\n"
                        "- Cron path: `crons/demo/x`\n" + task),
    ]
    for command in commands:
        items.append({
            "type": "function_call", "name": "bash", "call_id": "c%d" % len(items),
            "arguments": json.dumps({"command": command}),
        })
        items.append({
            "type": "function_call_output", "call_id": "c%d" % (len(items) - 1),
            "output": "ok\n",
        })
    items.append({"type": "message", "role": "assistant",
                  "content": [{"type": "output_text", "text": "done"}]})
    return {"model": model, "input": items, "tools": [
        {"type": "function", "name": "bash", "description": "run", "parameters": {}},
        {"type": "function", "name": "file_read", "description": "read", "parameters": {}},
    ]}


class ExplorerApiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "export").mkdir()
        rows = [
            trajectory("claude-opus-5", ["cd /work && cat a.txt", "grep -i x a.txt"],
                       "audit the alpha report"),
            trajectory("claude-sonnet-5", ["cd /work && cat b.txt", "ls -la"],
                       "summarise the beta thread"),
            trajectory("gpt-5.6-luna", ["cat c.txt"], "check the gamma cron"),
        ]
        (cls.root / "export" / "sample.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        cls.db = cls.root / "explorer" / "index.duckdb"
        ensure_index(cls.root, cls.db, progress=None)
        cls.client = TestClient(create_app(cls.db, STATIC_DIR))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def get(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200, "%s -> %s" % (path, response.text[:400]))
        return response.json()

    def test_every_read_endpoint_answers(self):
        trajectory_id = self.get("/api/trajectories?limit=1")["rows"][0]["trajectory_id"]
        other = self.get("/api/trajectories?limit=2")["rows"][1]["trajectory_id"]
        for path in [
            "/api/schema", "/api/overview", "/api/classes",
            "/api/classes?axis=workload", "/api/tools", "/api/calls?limit=5",
            "/api/trajectories", "/api/trajectories?model=claude-opus-5&sort=n_calls",
            "/api/trajectories/%s" % trajectory_id,
            "/api/trajectories/%s/similar" % trajectory_id,
            "/api/compare?a=%s&b=%s" % (trajectory_id, other),
            "/api/matched", "/api/matched?axes=harness,workload",
            "/api/sources", "/api/sources/rows?path=export/sample.jsonl",
            "/api/sources/row?path=export/sample.jsonl&line=1",
            "/api/functions", "/api/functions/bash",
            "/api/functions/bash?group_by=arg_shape",
            "/api/functions/bash/samples",
            "/api/functions/bash/samples?group_key=cat&per_model=2",
        ]:
            with self.subTest(path=path):
                self.get(path)

    def test_functions_group_by_command_and_split_cost(self):
        body = self.get("/api/functions")
        bash = next(row for row in body["tools"] if row["tool"] == "bash")
        self.assertEqual(bash["calls"], 5)
        self.assertEqual(bash["models"], 3)
        self.assertGreater(bash["spend_usd"], 0)

        detail = self.get("/api/functions/bash")
        verbs = {group["group_key"] for group in detail["groups"]}
        # `cd /work && cat a.txt` must group under cat, not under work.
        self.assertIn("cat", verbs)
        self.assertNotIn("work", verbs)
        cat = next(group for group in detail["groups"] if group["group_key"] == "cat")
        self.assertEqual(cat["calls"], 3)
        self.assertEqual(cat["models"], 3)
        for row in detail["by_model"]:
            self.assertIsNotNone(row["price_uncached"], row["model"])

    def test_samples_carry_arguments_and_cost_per_model(self):
        body = self.get("/api/functions/bash/samples?group_key=cat&per_model=1")
        self.assertEqual({entry["model"] for entry in body["models"]},
                         {"claude-opus-5", "claude-sonnet-5", "gpt-5.6-luna"})
        for entry in body["models"]:
            sample = entry["samples"][0]
            self.assertIn("cat", sample["arg_text"])
            self.assertGreater(sample["cost_share_usd"], 0)

    def test_unknown_tool_and_trajectory_are_404(self):
        self.assertEqual(self.client.get("/api/functions/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/trajectories/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
