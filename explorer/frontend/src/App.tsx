import { useState } from "react";
import { useApi } from "./api";
import { Classes, Tools } from "./views/Classes";
import { Functions } from "./views/Functions";
import { Compare } from "./views/Compare";
import { Overview } from "./views/Overview";
import { RawCalls } from "./views/RawCalls";
import { Sources } from "./views/Sources";
import { Filters, Trajectories } from "./views/Trajectories";
import { count } from "./format";

type Tab = "overview" | "classes" | "tools" | "trajectories" | "compare" | "sources" | "raw";

const TABS: { key: Tab; label: string; blurb: string }[] = [
  { key: "overview", label: "Overview", blurb: "what the corpus costs and why" },
  { key: "classes", label: "Classes", blurb: "how the inputs sort themselves" },
  { key: "tools", label: "Function calls", blurb: "same function, different models" },
  { key: "trajectories", label: "Trajectories", blurb: "browse runs, open one flow" },
  { key: "compare", label: "Compare", blurb: "align two runs, match cohorts" },
  { key: "sources", label: "Files", blurb: "browse export/ and results/ line by line" },
  { key: "raw", label: "Raw calls", blurb: "the unopinionated table" },
];

export function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [filters, setFilters] = useState<Filters>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [left, setLeft] = useState<string | null>(null);
  const [right, setRight] = useState<string | null>(null);
  const { data: schema } = useApi<any>("/api/schema");

  function openTrajectory(id: string) {
    setSelected(id);
    setTab("trajectories");
  }

  function compareWith(id: string) {
    if (!left || right) { setLeft(id); setRight(null); } else { setRight(id); }
    setTab("compare");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <h1>Trajectory Explorer</h1>
          <span className="subtitle">
            {schema
              ? <>{count(schema.calls)} reconstructed calls · {count(schema.trajectories)} trajectories · {schema.token_estimator}</>
              : "loading index…"}
          </span>
        </div>
        <div className="source-status" title="Every number on every tab comes from these files and nothing else">
          {(schema?.sources ?? []).map((source: any) => (
            <span className={`source ${source.kind}`} key={`${source.kind}:${source.path}`}>
              {source.path.split("/").pop()} · {count(source.line_count)} lines
            </span>
          ))}
          {schema && !schema.sources.length && <span className="source missing">no sources</span>}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((entry) => (
          <button
            key={entry.key}
            className={tab === entry.key ? "active" : ""}
            onClick={() => setTab(entry.key)}
            title={entry.blurb}
          >
            {entry.label}
            {entry.key === "compare" && (left || right) && (
              <span className="tab-badge">{[left, right].filter(Boolean).length}</span>
            )}
          </button>
        ))}
      </nav>

      <main className={tab === "raw" ? "main main-raw" : "main"}>
        {tab === "overview" && (
          <Overview onPickModel={(model) => { setFilters({ model }); setSelected(null); setTab("trajectories"); }} />
        )}
        {tab === "classes" && (
          <Classes onPickClass={(axis, value) => { setFilters({ [axis]: value }); setSelected(null); setTab("trajectories"); }} />
        )}
        {tab === "tools" && (
          <>
            <Functions onOpenTrajectory={openTrajectory} />
            <Tools />
          </>
        )}
        {tab === "trajectories" && (
          <Trajectories
            filters={filters}
            setFilters={setFilters}
            selected={selected}
            setSelected={setSelected}
            onCompare={compareWith}
          />
        )}
        {tab === "compare" && (
          <Compare left={left} right={right} setLeft={setLeft} setRight={setRight} />
        )}
        {tab === "sources" && <Sources onOpenTrajectory={openTrajectory} />}
        {tab === "raw" && <RawCalls />}
      </main>

      <footer className="app-foot">
        <span>
          Local only — reads <code>export/*.jsonl</code> and <code>results/*.jsonl</code>,
          makes no network requests.
        </span>
        <span>
          All token counts estimated (serialized chars ÷ 4). The export has no{" "}
          <code>usage</code> and no outcome labels: these are costs, not quality.
        </span>
        {selected && tab !== "trajectories" && (
          <button className="ghost small" onClick={() => openTrajectory(selected)}>
            back to {selected.slice(0, 8)}
          </button>
        )}
      </footer>
    </div>
  );
}
