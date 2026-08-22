import csv
import json

import compare


def write_routes_file(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, rec in enumerate(records):
            rec = dict(rec, trajectory=f"t{i}", switches=0)
            f.write(json.dumps(rec) + "\n")


SAMPLE_RECORDS = [
    {"n_calls": 2, "logged_model": "claude-opus-5", "route": ["claude-opus-5", "claude-sonnet-5"],
     "cost_logged_usd": 1.0, "cost_routed_usd": 0.6},
    {"n_calls": 2, "logged_model": "claude-opus-5", "route": ["claude-opus-5", "claude-opus-5"],
     "cost_logged_usd": 2.0, "cost_routed_usd": 2.0},
]


def test_discover_versions_finds_baseline_and_subfolders(tmp_path):
    results = tmp_path / "results"
    write_routes_file(results / "routes.jsonl", SAMPLE_RECORDS)
    write_routes_file(results / "v1_x" / "routes.jsonl", SAMPLE_RECORDS)
    write_routes_file(results / "v2_y" / "routes.jsonl", SAMPLE_RECORDS)
    (results / "notaversion").mkdir()  # no routes.jsonl inside -> must be skipped

    versions = compare.discover_versions(results)

    assert set(versions.keys()) == {"baseline", "v1_x", "v2_y"}
    assert versions["baseline"] == results / "routes.jsonl"
    assert versions["v1_x"] == results / "v1_x" / "routes.jsonl"


def test_summarize_reports_cost_and_quality_at_full_adoption(tmp_path):
    routes_path = tmp_path / "routes.jsonl"
    write_routes_file(routes_path, SAMPLE_RECORDS)

    summary = compare.summarize(routes_path)

    assert summary == {"cost_usd_full_adopt": 2.6, "quality_full_adopt": 0.75}


def test_write_comparison_csv_computes_delta_vs_baseline(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    summary = {
        "baseline": {"cost_usd_full_adopt": 10.0, "quality_full_adopt": 0.9},
        "v1_x": {"cost_usd_full_adopt": 8.0, "quality_full_adopt": 0.85},
    }

    out = compare.write_comparison_csv(results, summary)

    assert out.exists()
    with open(out, newline="") as f:
        rows = {r["version"]: r for r in csv.DictReader(f)}
    assert rows["baseline"]["cost_delta_vs_baseline_pct"] == "0.0"
    assert rows["v1_x"]["cost_delta_vs_baseline_pct"] == "-20.0"


def test_main_never_overwrites_baseline_frontier_csv(tmp_path, monkeypatch):
    results = tmp_path / "results"
    write_routes_file(results / "routes.jsonl", SAMPLE_RECORDS)
    write_routes_file(results / "v1_x" / "routes.jsonl", SAMPLE_RECORDS)
    sentinel = results / "frontier.csv"
    sentinel.write_text("do-not-touch\n")

    compare.main(results_root=results)

    assert sentinel.read_text() == "do-not-touch\n"
    assert (results / "v1_x" / "frontier.csv").exists()
    assert (results / "comparison.csv").exists()


def test_main_skips_malformed_version_without_crashing(tmp_path, capsys):
    results = tmp_path / "results"
    write_routes_file(results / "routes.jsonl", SAMPLE_RECORDS)
    (results / "v1_broken").mkdir(parents=True)
    (results / "v1_broken" / "routes.jsonl").write_text("not-json\n")

    compare.main(results_root=results)  # must not raise

    out = capsys.readouterr().out
    assert "v1_broken" in out
    with open(results / "comparison.csv", newline="") as f:
        rows = {r["version"] for r in csv.DictReader(f)}
    assert "v1_broken" not in rows
    assert "baseline" in rows


def test_experiments_csv_upsert_preserves_description_and_excludes_baseline(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    exp_path = results / "experiments.csv"
    with open(exp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["version", "description", "cost_usd_full_adopt",
                                           "quality_full_adopt", "cost_delta_vs_baseline_pct"])
        w.writeheader()
        w.writerow({"version": "v1_x", "description": "my first try",
                    "cost_usd_full_adopt": "999", "quality_full_adopt": "0", "cost_delta_vs_baseline_pct": "0"})

    summary = {
        "baseline": {"cost_usd_full_adopt": 10.0, "quality_full_adopt": 0.9},
        "v1_x": {"cost_usd_full_adopt": 8.0, "quality_full_adopt": 0.85},
    }
    compare.upsert_experiments_csv(results, summary)

    with open(exp_path, newline="") as f:
        rows = {r["version"]: r for r in csv.DictReader(f)}
    assert "baseline" not in rows
    assert rows["v1_x"]["description"] == "my first try"
    assert rows["v1_x"]["cost_usd_full_adopt"] == "8.0"
