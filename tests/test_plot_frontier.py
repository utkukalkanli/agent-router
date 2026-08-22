import csv
import json

from plot_frontier import frontier_rows


def sample_records():
    return [
        {"n_calls": 2, "logged_model": "claude-opus-5", "route": ["claude-opus-5", "claude-sonnet-5"],
         "cost_logged_usd": 1.0, "cost_routed_usd": 0.6},
        {"n_calls": 2, "logged_model": "claude-opus-5", "route": ["claude-opus-5", "claude-opus-5"],
         "cost_logged_usd": 2.0, "cost_routed_usd": 2.0},
    ]


def test_frontier_rows_sweeps_eleven_fractions_with_correct_endpoints():
    rows = frontier_rows(sample_records())

    assert len(rows) == 11
    assert [r["adopt_frac"] for r in rows] == [i / 10 for i in range(11)]

    # frac 0.0: nothing adopted -> full logged cost, quality stays at 1.0
    assert rows[0]["cost_usd"] == 3.0
    assert rows[0]["quality_placeholder"] == 1.0

    # frac 1.0: everything adopted -> full routed cost, quality drops per the placeholder signal
    assert rows[-1]["cost_usd"] == 2.6
    assert rows[-1]["quality_placeholder"] == 0.75


def test_plot_frontier_cli_writes_same_frontier_csv(tmp_path, monkeypatch):
    routes_path = tmp_path / "routes.jsonl"
    with open(routes_path, "w") as f:
        for rec in sample_records():
            rec = dict(rec, trajectory="t", switches=0)
            f.write(json.dumps(rec) + "\n")

    monkeypatch.setattr("sys.argv", ["plot_frontier.py", str(routes_path)])
    import plot_frontier
    plot_frontier.main()

    out_csv = tmp_path / "frontier.csv"
    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 11
    assert rows[0]["cost_usd"] == "3.0"
    assert rows[-1]["cost_usd"] == "2.6"
