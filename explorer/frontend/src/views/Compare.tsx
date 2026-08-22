import { useState } from "react";
import { CompareResult, Matched, useApi } from "../api";
import { BarRow, Note, Panel, Stat } from "../charts";
import { count, modelColor, pct, signedPct, signedUsd, tokens, usd } from "../format";

export function Compare(props: {
  left: string | null; right: string | null;
  setLeft: (id: string | null) => void; setRight: (id: string | null) => void;
}) {
  return (
    <>
      <PairCompare {...props} />
      <Cohorts />
    </>
  );
}

/** Two runs, aligned on what they DID rather than on what they said. */
function PairCompare(props: {
  left: string | null; right: string | null;
  setLeft: (id: string | null) => void; setRight: (id: string | null) => void;
}) {
  const { left, right } = props;
  const { data: similar } = useApi<any>(left ? `/api/trajectories/${left}/similar?limit=10` : null);
  const { data, error } = useApi<CompareResult>(
    left && right ? `/api/compare?a=${left}&b=${right}` : null,
  );

  if (!left) {
    return (
      <Note>
        Pick a trajectory from the <strong>Trajectories</strong> tab and press{" "}
        <em>compare</em> to anchor this view.
      </Note>
    );
  }

  return (
    <>
      <Panel
        title="Nearest runs to the anchor"
        help="Ranked on three things a router could see before the first token: the toolset it was handed, how deep the loop went, and the sequence of moves it made. Pick one to line the two up."
        right={<button className="ghost" onClick={() => { props.setLeft(null); props.setRight(null); }}>reset</button>}
      >
        {similar?.anchor && (
          <p className="anchor-line">
            anchor <span className="mono">{similar.anchor.trajectory_id}</span> ·
            {" "}<i className="swatch" style={{ background: modelColor(similar.anchor.model) }} />
            {similar.anchor.model} · {count(similar.anchor.n_calls)} calls ·
            {" "}{usd(similar.anchor.cost_usd)} · <span className="mono">{similar.anchor.shape_signature}</span>
          </p>
        )}
        {(similar?.neighbours ?? []).map((row: any) => (
          <BarRow
            key={row.trajectory_id}
            active={row.trajectory_id === right}
            label={
              <>
                <span className="mono">{row.trajectory_id}</span>
                <i className="swatch" style={{ background: modelColor(row.model) }} />
                {row.model}
              </>
            }
            sub={
              <>
                {count(row.n_calls)} calls · tool overlap {pct(row.tool_jaccard, 0)}
                {row.same_shape && <> · <strong>same loop shape</strong></>} ·
                {" "}<span className="mono">{row.shape_signature}</span>
              </>
            }
            value={row.similarity}
            max={1}
            text={`${usd(row.cost_usd)}  (${signedUsd(row.cost_delta_usd)})`}
            onClick={() => props.setRight(row.trajectory_id)}
          />
        ))}
        <p className="panel-foot warn">
          Similar is not exchangeable. These runs were not assigned to their models at
          random, so a cost gap between two neighbours is a mix of the model's price and
          whatever made the dispatcher pick it. Treat a single pair as an illustration,
          not an estimate.
        </p>
      </Panel>

      {error && <p className="fatal">{error}</p>}
      {data && <Alignment data={data} />}
    </>
  );
}

function Alignment(props: { data: CompareResult }) {
  const { left, right, steps, summary } = props.data;
  const maxStepCost = Math.max(
    ...steps.map((step) => Math.max(step.left?.cost_usd ?? 0, step.right?.cost_usd ?? 0)),
    1e-9,
  );

  return (
    <>
      <div className="stat-grid">
        <Stat
          label="Shared structure"
          value={pct(summary.shape_overlap, 0)}
          note={<>{count(summary.matched_steps)} of {count(summary.total_steps)} aligned steps are the same move</>}
        />
        <Stat
          label="Cost across matched steps"
          value={`${usd(summary.matched_cost_left)} → ${usd(summary.matched_cost_right)}`}
          note={<>{signedPct(summary.matched_cost_left ? summary.matched_cost_delta / summary.matched_cost_left : null)} on the part both runs did</>}
        />
        <Stat
          label="Cost in unmatched steps"
          value={`${usd(summary.unmatched_cost_left)} / ${usd(summary.unmatched_cost_right)}`}
          note="work only one of the two runs did — not comparable"
        />
        <Stat
          label="Total gap"
          value={signedUsd(summary.total_cost_delta)}
          note={summary.same_model ? "same model both sides — the gap is workload, not price" : "different models — price and workload are mixed together"}
        />
      </div>

      <Panel
        title="Step-by-step alignment"
        help="Longest-common-subsequence on what each call produced. Every export row opens with a unique user message, so two runs never share a literal prompt prefix — what they share is the shape of the loop."
      >
        <div className="align-head">
          <span>
            <i className="swatch" style={{ background: modelColor(left.trajectory.model) }} />
            <span className="mono">{left.trajectory.trajectory_id}</span> {left.trajectory.model} · {usd(left.trajectory.cost_usd)}
          </span>
          <span>step</span>
          <span>
            <i className="swatch" style={{ background: modelColor(right.trajectory.model) }} />
            <span className="mono">{right.trajectory.trajectory_id}</span> {right.trajectory.model} · {usd(right.trajectory.cost_usd)}
          </span>
        </div>
        <ol className="align">
          {steps.map((step, index) => (
            <li key={index} className={`align-row ${step.status}`}>
              <Side call={step.left} max={maxStepCost} side="left" />
              <span className="align-sig">
                <span className={`align-badge ${step.status}`}>
                  {step.status === "matched" ? "=" : step.status === "only_left" ? "◀" : "▶"}
                </span>
                <span className="mono">{step.signature}</span>
              </span>
              <Side call={step.right} max={maxStepCost} side="right" />
            </li>
          ))}
        </ol>
        <p className="panel-foot">
          <span className="align-badge matched">=</span> both runs made this move ·
          {" "}<span className="align-badge only_left">◀</span> only the left run ·
          {" "}<span className="align-badge only_right">▶</span> only the right run
        </p>
      </Panel>
    </>
  );
}

function Side(props: { call: any; max: number; side: "left" | "right" }) {
  if (!props.call) return <span className={`align-side ${props.side} empty`}>—</span>;
  const call = props.call;
  return (
    <span className={`align-side ${props.side}`}>
      <span className="align-cost">{usd(call.cost_usd)}</span>
      <span className="align-bar">
        <i style={{ width: `${(call.cost_usd / props.max) * 100}%` }} />
      </span>
      <span className="align-detail">
        #{call.call_index} · {tokens(call.est_input_tokens)} in ({pct(call.prefix_share, 0)} cached) ·
        {" "}{tokens(call.est_output_tokens)} out
        {call.tool_output_error && <span className="chip warn small">error</span>}
      </span>
    </span>
  );
}

const COHORT_AXES = [
  { key: "toolset_label,workload,size_band", label: "toolset · workload · size" },
  { key: "harness,workload,size_band,depth_band", label: "harness · workload · size · depth" },
  { key: "toolset_label,workload", label: "toolset · workload (coarse)" },
  { key: "cron_family,size_band", label: "cron family · size" },
];

/** The off-policy question, done by matching instead of by eyeballing. */
function Cohorts() {
  const [axes, setAxes] = useState(COHORT_AXES[0].key);
  const { data, error } = useApi<Matched>(`/api/matched?axes=${encodeURIComponent(axes)}`);
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Matching cohorts…</p>;
  const coverage = data.coverage;
  const maxSaving = Math.max(...data.cohorts.map((cohort) => cohort.saving_usd ?? 0), 1e-9);

  return (
    <Panel
      title="Matched cohorts — the off-policy estimate"
      help="Bucket trajectories on things a router could read before the first token, then read the per-model mean cost inside each bucket. Within a bucket the model that ran is treated as if it were assigned at random, so its mean estimates what the other runs in that bucket would have cost."
      right={
        <select value={axes} onChange={(event) => setAxes(event.target.value)}>
          {COHORT_AXES.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
        </select>
      }
    >
      <div className="stat-grid tight">
        <Stat label="Matched coverage" value={pct(coverage.share, 0)}
          note={<>{count(coverage.matched_trajectories)} of {count(coverage.total_trajectories)} runs sit in a bucket that saw 2+ models</>} />
        <Stat label="Actual cost, matched runs" value={usd(coverage.matched_actual_usd)} />
        <Stat label="If each bucket ran on its cheapest observed model" value={usd(coverage.matched_actual_usd - coverage.total_saving_usd)}
          tone="good"
          note={<>headroom {usd(coverage.total_saving_usd)} ({pct(coverage.total_saving_usd / coverage.matched_actual_usd, 0)})</>} />
      </div>

      <Note tone="warn">
        This is an upper bound, and it is not small. Assignment was <strong>not</strong>{" "}
        random: if Viktor already sent the hard work to expensive models, then inside a
        bucket the expensive model's runs <em>are</em> the hard ones, and this estimator
        reads that selection as if it were a price difference. The{" "}
        <code>size spread</code> column is the tell — the closer to 1.0, the more the
        models in that bucket really were seeing the same amount of work. It also assumes
        a cheaper model would have finished in the same number of calls, which is exactly
        the quality question the export cannot answer.
      </Note>

      <table className="grid">
        <thead>
          <tr>
            <th>cohort</th><th className="num">runs</th><th className="num">models</th>
            <th className="num">actual</th><th>cheapest observed</th>
            <th className="num">headroom</th><th className="num">size spread</th>
          </tr>
        </thead>
        <tbody>
          {data.cohorts.map((cohort) => (
            <tr key={cohort.cohort}>
              <td>{cohort.cohort}</td>
              <td className="num">{count(cohort.trajectories)}</td>
              <td className="num">{count(cohort.distinct_models)}</td>
              <td className="num">{usd(cohort.actual_cost_usd)}</td>
              <td>
                <i className="swatch" style={{ background: modelColor(cohort.cheapest_model) }} />
                {cohort.cheapest_model}
              </td>
              <td className="num cost-cell">
                <span className="cost-bar good" style={{ width: `${((cohort.saving_usd ?? 0) / maxSaving) * 100}%` }} />
                <strong>{usd(cohort.saving_usd)}</strong> <em>{pct(cohort.saving_pct, 0)}</em>
              </td>
              <td className={`num${(cohort.size_spread ?? 0) > 3 ? " warn-text" : ""}`}>
                {cohort.size_spread ? `${cohort.size_spread.toFixed(1)}×` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
