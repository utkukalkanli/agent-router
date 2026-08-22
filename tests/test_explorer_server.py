import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from explorer.index import ensure_index
from explorer.server import STATIC_DIR, create_app


class ExplorerServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        export = self.root / "export"
        export.mkdir()
        row = {
            "model": "gpt-5.6-luna",
            "input": [
                {"type": "message", "role": "system", "content": "system"},
                {"type": "message", "role": "user", "content": "task"},
            ],
            "tools": [],
        }
        (export / "sample.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.db = self.root / "explorer" / "index.duckdb"
        ensure_index(self.root, self.db, progress=None)
        self.client = TestClient(create_app(self.db, STATIC_DIR))

    def tearDown(self):
        self.temp.cleanup()

    def test_schema_reports_sources_and_counts(self):
        response = self.client.get("/api/schema")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["calls"], body["trajectories"]), (1, 1))
        self.assertEqual(body["source_kinds"], ["export"])
        self.assertEqual(body["sources"][0]["path"], "export/sample.jsonl")

    def test_calls_select_sort_paginate_and_never_return_raw(self):
        response = self.client.get(
            "/api/calls",
            params={
                "limit": 1, "offset": 0, "sort": "est_input_tokens",
                "order": "desc", "columns": "trajectory_id,model,output_kind",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(set(body["rows"][0]), {
            "call_id", "trajectory_id", "model", "output_kind",
        })
        self.assertEqual(body["rows"][0]["output_kind"], "unobservable")
        self.assertNotIn("raw_json", body["rows"][0])

    def test_calls_reject_unknown_columns_and_limits_over_500(self):
        self.assertEqual(
            self.client.get("/api/calls", params={"sort": "drop table calls"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.get("/api/calls", params={"columns": "raw_json"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.get("/api/calls", params={"limit": 501}).status_code,
            422,
        )

    def test_built_frontend_is_served_same_origin(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Viktor Trajectory Explorer", response.text)


if __name__ == "__main__":
    unittest.main()
