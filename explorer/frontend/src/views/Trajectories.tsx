import { useMemo, useState } from "react";
import { TrajectoryList, useApi } from "../api";
import { Note } from "../charts";
import { count, modelColor, pct, tokens, usd } from "../format";
import { Flow } from "./Flow";

export type Filters = Record<string, string | undefined>;

const FACETS: { key: string; label: string }[] = [
  { key: "model", label: "Model" },
  { key: "workload", label: "Workload" },
  { key: "trigger_kind", label: "Trigger" },
  { key: "harness", label: "Harness" },
  { key: "surface", label: "Surface" },
  { key: "size_band", label: "Size" },
  { key: "depth_band", label: "Depth" },
  { key: "toolset_label", label: "Toolset" },
  { key: "shape_signature", label: "Loop shape" },
  { key: "cron_family", label: "Cron family" },
];
const SORTS: { key: string; label: string }[] = [
  { key: "cost_usd", label: "cost" },
  { key: "n_calls", label: "calls" },
  { key: "billed_input_tokens", label: "input tokens" },
  { key: "output_tokens", label: "output tokens" },
  { key: "cache_hit_rate", label: "cache hit rate" },
  { key: "saving_usd", label: "routing saving" },
  { key: "started_at", label: "start time" },
];

export function Trajectories(props: {
  filters: Filters;
  setFilters: (next: Filters) => void;
  selected: string | null;
  setSelected: (id: string | null) => void;
  onCompare: (id: string) => void;
}) {
  const [sort, setSort] = useState("cost_usd");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");

  const path = useMemo(() => {
    const params = new URLSearchParams({ sort, order, limit: "150" });
    for (const [key, value] of Object.entries(props.filters)) {
      if (value) params.set(key, value);
    }
    if (search) params.set("q", search);
    return `/api/trajectories?${params}`;
  }, [props.filters, sort, order, search]);

  const { data, error, loading } = useApi<TrajectoryList>(path);
  const active = Object.entries(props.filters).filter(([, value]) => value);

  if (props.selected) {
    return (
      <Flow
        trajectoryId={props.selected}
        onCompare={props.onCompare}
        onClose={() => props.setSelected(null)}
      />
    );
  }
  if (error) return <p className="fatal">{error}</p>;

  const maxCost = Math.max(...(data?.rows ?? []).map((row) => row.cost_usd ?? 0), 0);

  return (
    <>
      <div className="filter-bar">
        <form
          onSubmit={(event) => { event.preventDefault(); setSearch(query); }}
        >
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search opening message, cron path, tool names, id…"
          />
        </form>
        <label>
          sort
          <select value={sort} onChange={(event) => setSort(event.target.value)}>
            {SORTS.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
          </select>
        </label>
        <button className="ghost" onClick={() => setOrder(order === "desc" ? "asc" : "desc")}>
          {order === "desc" ? "↓ high first" : "↑ low first"}
        </button>
        {(active.length > 0 || search) && (
          <button className="ghost" onClick={() => { props.setFilters({}); setSearch(""); setQuery(""); }}>
            clear all
          </button>
        )}
      </div>

      {active.length > 0 && (
        <p className="active-filters">
          {active.map(([key, value]) => (
            <button key={key} className="chip removable" onClick={() => props.setFilters({ ...props.filters, [key]: undefined })}>
              {key}: {value} ×
            </button>
          ))}
        </p>
      )}

      <div className="facets">
        {FACETS.map((facet) => {
          const options = data?.facets?.[facet.key] ?? [];
          if (options.length < 2) return null;
          return (
            <label key={facet.key} className="facet">
              <span>{facet.label}</span>
              <select
                value={props.filters[facet.key] ?? ""}
                onChange={(event) =>
                  props.setFilters({ ...props.filters, [facet.key]: event.target.value || undefined })
                }
              >
                <option value="">all</option>
                {options.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.value} ({option.n})
                  </option>
                ))}
              </select>
            </label>
          );
        })}
      </div>

      {data && (
        <p className="result-summary">
          <strong>{count(data.summary.trajectories)}</strong> trajectories ·
          {" "}{count(data.summary.calls)} calls ·
          {" "}<strong>{usd(data.summary.cost_usd)}</strong> logged ·
          {data.summary.routed_cost_usd != null && (
            <> {usd(data.summary.routed_cost_usd)} rerouted ·</>
          )}
          {" "}cache {pct(data.summary.cache_hit_rate, 0)}
          {loading && <span className="loading-inline"> · querying…</span>}
        </p>
      )}

      <Note>
        Ranked by {SORTS.find((option) => option.key === sort)?.label}. Click a row for its
        call-by-call flow; the compare button on that page lines two runs up side by side.
      </Note>

      <table className="grid rows">
        <thead>
          <tr>
            <th>trajectory</th><th>model</th><th className="num">calls</th>
            <th className="num">input</th><th className="num">output</th>
            <th className="num">cache</th><th>class</th><th className="num">cost</th><th />
          </tr>
        </thead>
        <tbody>
          {(data?.rows ?? []).map((row) => (
            <tr key={row.trajectory_id} onClick={() => props.setSelected(row.trajectory_id)}>
              <td className="mono">{row.trajectory_id}</td>
              <td className="nowrap">
                <i className="swatch" style={{ background: modelColor(row.model) }} />
                {row.model}
              </td>
              <td className="num">
                {count(row.n_calls)}
                {row.n_error_calls > 0 && (
                  <span
                    className="err-frac"
                    title={`${row.n_error_calls} of ${row.n_calls} calls saw a tool result that reads like an error`}
                  >
                    {row.n_error_calls}⚠
                  </span>
                )}
              </td>
              <td className="num">{tokens(row.billed_input_tokens)}</td>
              <td className="num">{tokens(row.total_est_output_tokens)}</td>
              <td className="num">{pct(row.cache_hit_rate, 0)}</td>
              <td className="classes">
                <span className="chip">{row.workload}</span>
                <span className="chip">{row.size_band}</span>
                <span className="chip mono">{row.shape_signature}</span>
              </td>
              <td className="num cost-cell">
                <span className="cost-bar" style={{ width: `${maxCost ? ((row.cost_usd ?? 0) / maxCost) * 100 : 0}%` }} />
                <strong>{usd(row.cost_usd)}</strong>
              </td>
              <td>
                <button
                  className="ghost small"
                  onClick={(event) => { event.stopPropagation(); props.onCompare(row.trajectory_id); }}
                >
                  compare
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data && data.rows.length >= data.limit && (
        <p className="panel-foot">
          Showing the top {data.limit} of {count(data.summary.trajectories)} — narrow with a
          filter rather than paging; the tail is cheap by construction.
        </p>
      )}
    </>
  );
}
