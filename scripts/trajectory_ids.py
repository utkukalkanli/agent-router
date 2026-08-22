#!/usr/bin/env python3
"""Assign stable trajectory ids and surface the metadata hidden in the opening message.

The export ships no `trajectory_id`, and grouping on the opening messages (what
`load_trajectories.py` does) finds nothing in the real chunk: every row has a unique
first user message and the item-level prefix two rows share is always exactly 1 item
(the system message). One row = one complete trajectory.

But the rows are not anonymous. Viktor's first user message embeds a `# === Thread info ===`
block, and the trailing cron memory carries a `Cron path:`. Between them they yield:

  Path:             the thread this run belongs to      (786/1000 rows)
  Triggered by:     cron / trigger / a platform user id  (790/1000)
  Current datetime: a real wall-clock timestamp          (786/1000)
  Cron path:        the cron identity                    (779/1000, 351 distinct)

The timestamps matter: the brief says the export has no timing. It does, in the prompt text.

THE REDACTION TRAP: placeholders are only consistent *within* a trajectory. A path that
redacts to `PII_PATH_2` is a different real path in every row it appears in — grouping on
it merges 35 unrelated trajectories. Same for thread names collapsed to `PII_PHONE_NUMBER_1`.
This script keys on (tenant, path) and flags collapsed paths instead of trusting them.

Usage: python scripts/trajectory_ids.py export/trajectories_v1_01.jsonl
Writes results/trajectory_index.jsonl — one record per row, joinable on `row`.
"""
import json, re, sys, hashlib, collections
from pathlib import Path

PATH_RE = re.compile(r'^\s*-\s*Path:\s*`([^`]+)`', re.M)
TRIG_RE = re.compile(r'^\s*-\s*Triggered by:\s*`([^`]+)`', re.M)
DT_RE   = re.compile(r'^\s*-\s*Current datetime:\s*`([^`]+)`', re.M)
CRON_RE = re.compile(r'^\s*-\s*Cron path:\s*`([^`]+)`', re.M)
ISO_RE  = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})')
PII_RE  = re.compile(r'PII_[A-Z_]+_\d+')


def sha(obj, n=10):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:n]


def first_user_text(req):
    """Viktor packs the whole task briefing into input[1]; it is a <system> wrapper."""
    for item in req["input"]:
        if item.get("role") == "user":
            c = item.get("content")
            if isinstance(c, str):
                return c
            return " ".join(p.get("text", "") for p in c
                            if isinstance(p, dict) and p.get("type") == "input_text")
    return ""


def path_is_collapsed(path):
    """True if redaction destroyed what made this path unique across trajectories."""
    if path is None:
        return False
    leaf = path.rstrip("/").split("/")[-1]
    return bool(PII_RE.fullmatch(path.lstrip("/")) or PII_RE.search(leaf))


def extract(req):
    t = first_user_text(req)
    def one(rx):
        m = rx.search(t)
        return m.group(1) if m else None
    dt = one(DT_RE)
    iso = ISO_RE.match(dt).group(1) if dt and ISO_RE.match(dt) else None
    return {
        "tenant": sha(req["input"][0]),          # system prompt identifies the workspace
        "thread_path": one(PATH_RE),
        "triggered_by": one(TRIG_RE),
        "datetime_raw": dt,
        "timestamp": iso,
        "timezone": dt.split(" in ")[-1] if dt and " in " in dt else None,
        "cron_path": one(CRON_RE),
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "export/trajectories_v1_01.jsonl"
    rows = [json.loads(l) for l in open(src) if l.strip()]

    recs = []
    for i, r in enumerate(rows):
        meta = extract(r)
        collapsed = path_is_collapsed(meta["thread_path"])
        # Stable id: the content hash of the row is always unique and always defined.
        # thread_key is the *joinable* one — None when redaction made it meaningless.
        recs.append({
            "row": i,
            "trajectory_id": sha(r["input"], 12),
            "thread_key": None if (collapsed or not meta["thread_path"])
                          else f'{meta["tenant"]}:{meta["thread_path"]}',
            "path_collapsed": collapsed,
            "model": r["model"],
            "family": "claude" if r["model"].startswith("claude") else "gpt",
            "n_items": len(r["input"]),
            **meta,
        })

    out = Path("results"); out.mkdir(exist_ok=True)
    dest = out / "trajectory_index.jsonl"
    with open(dest, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

    # ---- report ----
    n = len(recs)
    have = lambda k: sum(1 for r in recs if r[k])
    print(f"rows={n}  distinct trajectory_id={len({r['trajectory_id'] for r in recs})}  "
          f"(no two rows are the same trajectory)")
    print(f"metadata recovered — thread_path {have('thread_path')}/{n}  "
          f"timestamp {have('timestamp')}/{n}  cron_path {have('cron_path')}/{n}  "
          f"triggered_by {have('triggered_by')}/{n}")
    print(f"tenants (distinct system prompts): {len({r['tenant'] for r in recs})}")
    print(f"paths discarded as redaction-collapsed: {sum(1 for r in recs if r['path_collapsed'])}")

    ts = sorted(r["timestamp"] for r in recs if r["timestamp"])
    if ts:
        print(f"time span: {ts[0]} -> {ts[-1]}   "
              f"({collections.Counter(t[:10] for t in ts).most_common(1)[0][1]} rows on the modal day)")

    keyed = collections.defaultdict(list)
    for r in recs:
        if r["thread_key"]:
            keyed[r["thread_key"]].append(r)
    dupes = {k: v for k, v in keyed.items() if len(v) > 1}
    print(f"\nsame-thread groups on the clean key: {len(dupes)} "
          f"({sum(len(v) for v in dupes.values())} rows)")
    for k, v in sorted(dupes.items(), key=lambda x: -len(x[1])):
        print(f"  {k}")
        for r in sorted(v, key=lambda r: r["timestamp"] or ""):
            print(f"      row {r['row']:4}  {r['model']:18} items={r['n_items']:4}  {r['datetime_raw']}")

    # what naive path-only grouping would have merged
    bypath = collections.defaultdict(set)
    for r in recs:
        if r["thread_path"]:
            bypath[r["thread_path"]].add(r["tenant"])
    cross = sum(1 for p, t in bypath.items() if len(t) > 1)
    print(f"\npaths seen under >1 tenant (would merge wrongly without the tenant key): {cross}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
