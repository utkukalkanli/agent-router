import { Overview as OverviewData, useApi } from "../api";
import { BarRow, ColumnChart, Note, NumTh, Panel, Stat, StackedBar, Th } from "../charts";
import { count, modelColor, pct, signedPct, tokens, usd } from "../format";
import { SOURCE_HELP } from "../glossary";

export function Overview(props: { onPickModel: (model: string) => void }) {
  const { data, error, loading } = useApi<OverviewData>("/api/overview");
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">{loading ? "Aggregating…" : ""}</p>;

  const h = data.headline;
  const maxModelCost = Math.max(...data.by_model.map((m) => m.cost_usd ?? 0));
  const firstCall = data.by_call_index.find((row) => row.call_index === 0);
  const coldShare = firstCall && h.logged_cost_usd ? firstCall.cost_usd / h.logged_cost_usd : null;
  const topDecile = data.concentration[0];
  const topShare = topDecile && h.logged_cost_usd ? topDecile.cost_usd / h.logged_cost_usd : null;

  return (
    <>
      <div className="stat-grid">
        <Stat
          label="Estimated spend, logged route"
          value={usd(h.logged_cost_usd)}
          note={<>{usd(h.input_cost_usd)} input + {usd(h.output_cost_usd)} output</>}
        />
        <Stat
          label="Calls that actually ran"
          value={count(h.calls)}
          note={<>in {count(h.trajectories)} trajectories · {(h.calls / h.trajectories).toFixed(1)} per run</>}
        />
        <Stat
          label="Prefix cache hit rate"
          value={pct(h.cache_hit_rate)}
          note={<>{tokens(h.cached_tokens)} of {tokens(h.billed_input_tokens)} input tokens re-read</>}
        />
        {data.route_join?.healthy ? (
          <Stat
            label="Cost if rerouted"
            value={usd(h.routed_cost_usd)}
            tone={h.routed_delta_pct < 0 ? "good" : undefined}
            note={<>{signedPct(h.routed_delta_pct)} vs logged · results/routes.jsonl</>}
          />
        ) : (
          <Stat
            label="Spend in the top 10% of runs"
            value={usd(topDecile?.cost_usd)}
            note={<>{pct(topShare, 0)} of everything · {count(topDecile?.trajectories)} trajectories</>}
          />
        )}
      </div>

      <Note>
        Every token count on this page is an <strong>estimate</strong> — serialized JSON
        chars ÷ 4. The export ships no <code>usage</code> field, so nothing here is measured.
        Outputs are recovered from the next call's input, which leaves {count(h.unobservable_calls)}{" "}
        final turns ({pct(h.unobservable_calls / h.calls, 0)} of calls) with no observable output at all.
      </Note>

      <RouteJoin join={data.route_join} />

      <Panel
        title="Where the money goes"
        help="Spend by the model that actually served the call. One model serves a whole trajectory, so this is also a picture of which models get handed which work."
      >
        {data.by_model.map((row) => (
          <BarRow
            key={row.model}
            label={row.model}
            sub={<>{count(row.trajectories)} runs · {count(row.calls)} calls · ${row.uncached}/{row.cached} per 1M in</>}
            value={row.cost_usd ?? 0}
            max={maxModelCost}
            color={modelColor(row.model)}
            text={`${usd(row.cost_usd)} · ${pct((row.cost_usd ?? 0) / h.logged_cost_usd, 0)}`}
            onClick={() => props.onPickModel(row.model)}
          />
        ))}
        <p className="panel-foot">Click a model to see its trajectories.</p>
      </Panel>

      <div className="panel-row">
        <Panel
          title="The cold start is the line item"
          help="Cost by position in the agent loop. Call 0 has no cache to hit and pays the uncached rate for the entire prompt; every later call re-reads most of it."
        >
          <ColumnChart
            height={170}
            data={data.by_call_index.map((row) => ({
              key: row.call_index === 20 ? "20+" : String(row.call_index),
              value: row.cost_usd ?? 0,
              color: row.call_index === 0 ? "var(--c-cold)"
                : row.call_index === 20 ? "var(--accent-dim)" : "var(--accent)",
              title: `call ${row.call_index}: ${usd(row.cost_usd)} across ${count(row.calls)} calls, mean prefix share ${pct(row.mean_prefix_share)}`,
            }))}
            format={usd}
          />
          <p className="panel-foot">
            First call of each run: <strong>{pct(coldShare, 0)}</strong> of all spend. The
            dimmed last column pools every call from 20 onward, so it is a total, not a
            position.
          </p>
        </Panel>

        <Panel
          title="Spend is concentrated"
          help="Trajectories sorted by cost, split into deciles. A router that only changes the cheap tail cannot move the total."
        >
          <ColumnChart
            height={170}
            cumulative
            data={data.concentration.map((row) => ({
              key: `D${row.decile}`,
              value: row.cost_usd ?? 0,
              color: row.decile === 1 ? "var(--c-cold)" : "var(--accent-dim)",
              title: `decile ${row.decile}: ${count(row.trajectories)} runs, ${usd(row.cost_usd)}`,
            }))}
            format={usd}
          />
          <p className="panel-foot">
            Top 10% of runs: <strong>{pct(topShare, 0)}</strong> of all spend. The line is the
            cumulative share.
          </p>
        </Panel>
      </div>

      <div className="panel-row">
        <Panel
          title="Token mix"
          help="What the money is actually buying. Cached input is re-read prompt, billed at the cache-read rate — a tenth of the uncached rate on every model in this log."
        >
          <StackedBar
            height={22}
            segments={[
              { key: "cached", value: h.cached_tokens, color: "var(--c-cached)", label: "cached input" },
              { key: "new", value: h.billed_input_tokens - h.cached_tokens, color: "var(--c-new)", label: "uncached input" },
              { key: "output", value: h.output_tokens, color: "var(--c-out)", label: "recovered output" },
            ]}
          />
          <ul className="legend">
            <li><i style={{ background: "var(--c-cached)" }} />cached input {tokens(h.cached_tokens)}</li>
            <li><i style={{ background: "var(--c-new)" }} />uncached input {tokens(h.billed_input_tokens - h.cached_tokens)}</li>
            <li><i style={{ background: "var(--c-out)" }} />recovered output {tokens(h.output_tokens)}</li>
          </ul>
          <p className="panel-foot">
            {pct(h.cached_tokens / h.billed_input_tokens, 0)} of input tokens are cache re-reads.
            At a tenth of the uncached rate they contribute roughly{" "}
            {pct((h.cached_tokens * 0.1) / (h.cached_tokens * 0.1 + (h.billed_input_tokens - h.cached_tokens)), 0)}{" "}
            of the input bill — that is the whole point of the cache, and the whole risk in
            switching models mid-run.
          </p>
        </Panel>

        <Panel
          title="What the model produced"
          help="Every call is classified by what came back: a single tool call, several at once, an apply_patch, prose, or — for the last turn of every run — nothing observable."
        >
          {data.by_output_kind.map((row) => (
            <BarRow
              key={row.output_kind}
              label={row.output_kind}
              sub={<>{count(row.calls)} calls · mean {tokens(row.mean_output_tokens)} output tokens</>}
              value={row.cost_usd ?? 0}
              max={Math.max(...data.by_output_kind.map((r) => r.cost_usd ?? 0))}
              color={`var(--c-${row.output_kind === "multi_tool" ? "multi" : row.output_kind === "tool_call" ? "tool" : row.output_kind === "patch" ? "patch" : row.output_kind === "text" ? "text" : "unobs"})`}
              text={usd(row.cost_usd)}
            />
          ))}
        </Panel>
      </div>

      <Panel title="What this page reads" help="Nothing else. No network, no telemetry, no other files.">
        <table className="grid">
          <thead>
            <tr>
              <Th help={SOURCE_HELP.kind}>kind</Th>
              <Th help={SOURCE_HELP.file}>file</Th>
              <NumTh help={SOURCE_HELP.lines}>lines</NumTh>
              <NumTh help={SOURCE_HELP.parseErrors}>parse errors</NumTh>
              <NumTh help={SOURCE_HELP.size}>size</NumTh>
            </tr>
          </thead>
          <tbody>
            {data.sources.map((source) => (
              <tr key={`${source.kind}:${source.path}`}>
                <td><span className={`tag ${source.kind}`}>{source.kind}</span></td>
                <td className="mono">{source.path}</td>
                <td className="num">{count(source.line_count)}</td>
                <td className="num">{count(source.parse_errors)}</td>
                <td className="num">{(source.size_bytes / 1e6).toFixed(1)} MB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}


/** Whether results/*.jsonl attached to the reconstructed calls — and what to do if not. */
function RouteJoin(props: { join: any }) {
  const join = props.join;
  if (!join) return null;
  if (!join.present) {
    return (
      <Note>
        No router output loaded. Drop a <code>results/*.jsonl</code> with one record per
        trajectory (<code>trajectory</code>, <code>route</code>) and the routed-cost columns
        light up across every tab.
      </Note>
    );
  }
  if (join.healthy) {
    return (
      <Note>
        Router output joined: {count(join.joined_calls)} calls carry a routing decision from{" "}
        {count(join.matched_trajectories)} matched trajectories.
      </Note>
    );
  }
  return (
    <Panel
      title="Router output is loaded but does not join"
      help="This is why every route column on every tab reads blank. It is a real finding about the artifact, not a UI bug."
    >
      <p className="panel-body">
        <code>results/routes.jsonl</code> carries {count(join.result_trajectory_ids)} trajectory
        ids. <strong>{count(join.matched_trajectories)}</strong> of them match a trajectory in
        the export, so {count(join.joined_calls)} calls carry a routing decision.
      </p>
      <div className="id-compare">
        <div>
          <span className="id-label">ids in results/routes.jsonl</span>
          {join.sample_unmatched_ids.map((id: string) => <code key={id}>{id}</code>)}
        </div>
        <div>
          <span className="id-label">ids the reconstructed calls use</span>
          {join.sample_expected_ids.map((id: string) => <code key={id}>{id}</code>)}
        </div>
      </div>
      <p className="panel-foot warn">
        The two sides disagree about what a trajectory <em>is</em>.{" "}
        <code>scripts/load_trajectories.py</code>, which the baseline router uses, assumes one
        export row is one LLM call and groups rows by their opening message. On this export
        every row already opens with a unique user message, so that grouping returns mostly
        singletons. <code>scripts/decompose.py</code> — what this explorer indexes — reads one
        row as one complete trajectory and unfolds the agent loop inside it into the requests that actually ran. Different unit, different id,
        empty join.
      </p>
      <p className="panel-foot">
        Everything else on every tab is computed straight from <code>export/*.jsonl</code> and
        is unaffected. To make the route columns work, a router needs to emit ids minted the
        same way <code>decompose.py</code> mints them.
      </p>
    </Panel>
  );
}
