import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def demo_data():
    source = (DOCS / "data.js").read_text(encoding="utf-8")
    match = re.fullmatch(r"window\.ROUTEMAXXING_DEMO = (.*);\n", source, re.S)
    assert match
    return json.loads(match.group(1))


def test_static_demo_has_no_absolute_asset_paths_and_all_assets_exist():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assets = re.findall(r'(?:href|src)="(\./[^"]+)"', html)
    assert {"./styles.css", "./data.js", "./app.js"} <= set(assets)
    assert all((DOCS / asset[2:]).is_file() for asset in assets)
    assert 'href="/' not in html
    assert 'src="/' not in html


def test_demo_data_is_aggregate_only_and_matches_locked_headline():
    data = demo_data()
    encoded = json.dumps(data)
    assert data["headline"]["routed_cost"] == 48.1798
    assert data["headline"]["savings_pct"] == 45.79
    assert data["headline"]["failure_upper95_pp"] <= 2.0
    assert data["support"]["gpt-5.6-luna"]["eligible"] is False
    assert "trajectory_id" not in encoded
    assert "opening_user" not in encoded
    assert "<PERSON" not in encoded


def test_certified_demo_routes_preserve_family_and_never_upgrade():
    data = demo_data()
    pricing = data["pricing"]
    for default, selected in data["certified_downgrades"].items():
        assert default.split("-")[0] == selected.split("-")[0]
        assert all(a <= b for a, b in zip(pricing[selected], pricing[default]))


def test_pages_workflow_deploys_docs_directory():
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "path: docs" in workflow
