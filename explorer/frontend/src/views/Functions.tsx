import { useState } from "react";
import { Json, useApi } from "../api";
import { Note, NumTh, Panel, Th } from "../charts";
import { count, modelColor, pct, tokens, usd } from "../format";
import { FUNCTION_HELP, FUNCTION_MODEL_HELP } from "../glossary";

/** Cost per call is tiny; milli-dollars keep the column readable. */
function milli(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 1000).toFixed(2)}m$` : "—";
}

/** Hold the function fixed, let the model vary. */
export function Functions(props: { onOpenTrajectory: (id: string) => void }) {
  const [tool, setTool] = useState<string | null>(null);
  const [groupBy, setGroupBy] = useState<"command_verb" | "arg_shape">("command_verb");
  const [group, setGroup] = useState<string | null>(null);
  const { data, error } = useApi<{ tools: Json[]; totals: Json }>("/api/functions");

  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Grouping function calls…</p>;

  return (
    <>
      <Note>
        Every function call the models actually emitted — {count(data.totals.calls)} of them —
        grouped by the function they called. Within a group the job is held roughly fixed and
        the model varies, so the columns are comparable in a way whole trajectories never are.
        Cost is the request that produced the call; when one request emitted several calls at
        once its cost is split evenly across them, which is an attribution, not a measurement.
      </Note>

      <Panel
        title="Functions, by who called them and what they paid"
        help="Click a function to break it into comparable groups. `cost spread` is the ratio between the most and least expensive model per call of that function — the crude ceiling on what routing this one function could save."
      >
        <table className="grid rows">
          <thead>
            <tr>
              <Th help={FUNCTION_HELP.function}>function</Th>
              <NumTh help={FUNCTION_HELP.calls}>calls</NumTh>
              <NumTh help={FUNCTION_HELP.models}>models</NumTh>
              <NumTh help={FUNCTION_HELP.args}>args</NumTh>
              <NumTh help={FUNCTION_HELP.result}>result</NumTh>
              <NumTh help={FUNCTION_HELP.errors}>errors</NumTh>
              <NumTh help={FUNCTION_HELP.costPerCall}>$/call</NumTh>
              <NumTh help={FUNCTION_HELP.spend}>spend</NumTh>
              <NumTh help={FUNCTION_HELP.spread}>spread</NumTh>
            </tr>
          </thead>
          <tbody>
            {data.tools.map((row) => (
              <tr
                key={row.tool}
                className={tool === row.tool ? "selected" : ""}
                onClick={() => { setTool(tool === row.tool ? null : row.tool); setGroup(null); }}
              >
                <td className="mono">{row.tool}</td>
                <td className="num">{count(row.calls)}</td>
                <td className="num">{count(row.models)}</td>
                <td className="num">{tokens(row.median_arg_tokens)}</td>
                <td className="num">{tokens(row.median_result_tokens)}</td>
                <td className={`num${row.error_rate > 0.2 ? " warn-text" : ""}`}>{pct(row.error_rate, 0)}</td>
                <td className="num">{milli(row.mean_cost_usd)}</td>
                <td className="num">{usd(row.spend_usd)}</td>
                <td className={`num${(row.cost_spread ?? 0) > 3 ? " warn-text" : ""}`}>
                  {row.cost_spread ? `${row.cost_spread.toFixed(1)}×` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="panel-foot">
          `args` and `result` are medians in estimated tokens. A high `errors` rate means the
          tool's output text matched an error pattern — it is a keyword match, not a status code.
        </p>
      </Panel>

      {tool && (
        <FunctionDetail
          tool={tool}
          groupBy={groupBy}
          setGroupBy={(next) => { setGroupBy(next); setGroup(null); }}
          group={group}
          setGroup={setGroup}
          onOpenTrajectory={props.onOpenTrajectory}
        />
      )}
    </>
  );
}

function FunctionDetail(props: {
  tool: string;
  groupBy: "command_verb" | "arg_shape";
  setGroupBy: (next: "command_verb" | "arg_shape") => void;
  group: string | null;
  setGroup: (next: string | null) => void;
  onOpenTrajectory: (id: string) => void;
}) {
  const { data, error } = useApi<Json>(
    `/api/functions/${encodeURIComponent(props.tool)}?group_by=${props.groupBy}`,
  );
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Loading {props.tool}…</p>;

  return (
    <>
      <Panel
        title={`${props.tool} — by model`}
        help="Every call to this one function, split by the model that emitted it. Same function, so differences in argument size and result size are differences in how the models use it."
      >
        <ModelTable rows={data.by_model} />
        <p className="panel-foot">
          Ratios are against the cheapest-per-call model in the table.{" "}
          <strong>observed ×</strong> is what it actually cost;{" "}
          <strong>list price ×</strong> is what the rate card alone predicts;{" "}
          <strong>behaviour ×</strong> is what is left — above 1.0 the model reaches this
          function with a bigger prompt or writes more when it gets there, below 1.0 it is
          leaner than its price suggests. Highlighted past ±20%.
        </p>
      </Panel>

      <Panel
        title={`${props.tool} — comparable groups`}
        help={props.groupBy === "command_verb"
          ? "Grouped by the executable the command ran, looking past `cd x &&` prefixes. A heuristic: a pipeline is reduced to its first stage."
          : "Grouped by which argument keys the call set — the call's signature, ignoring its values."}
        right={
          <select
            value={props.groupBy}
            onChange={(event) => props.setGroupBy(event.target.value as "command_verb" | "arg_shape")}
          >
            <option value="command_verb">group by command</option>
            <option value="arg_shape">group by argument keys</option>
          </select>
        }
      >
        <p className="panel-foot" style={{ marginTop: 0, marginBottom: 10 }}>
          {count(data.comparable_groups)} of {count(data.groups.length)} groups were used by
          more than one model — those are the ones worth comparing.
        </p>
        <div className="group-chips">
          {data.groups.map((row: Json) => (
            <button
              key={row.group_key}
              className={`group-chip${props.group === row.group_key ? " active" : ""}${row.models > 1 ? "" : " single"}`}
              onClick={() => props.setGroup(props.group === row.group_key ? null : row.group_key)}
              title={`${row.calls} calls · ${row.models} model${row.models === 1 ? "" : "s"} · ${usd(row.spend_usd)}`}
            >
              <span className="mono">{row.group_key}</span>
              <em>{count(row.calls)}</em>
              <i>{row.models}m</i>
            </button>
          ))}
        </div>
        {props.group && (
          <div className="group-detail">
            <ModelTable
              rows={(data.groups.find((row: Json) => row.group_key === props.group)?.by_model) ?? []}
            />
          </div>
        )}
      </Panel>

      {props.group && (
        <Samples
          tool={props.tool}
          groupBy={props.groupBy}
          groupKey={props.group}
          onOpenTrajectory={props.onOpenTrajectory}
        />
      )}
    </>
  );
}

/** Separates the two reasons one model costs more than another on the same function:
 *  it is priced higher, or it uses the function more expensively (longer prompts by the
 *  time it gets there, bigger arguments, more output). The list-price ratio is the
 *  first; whatever is left over is the second. */
function ModelTable(props: { rows: Json[] }) {
  const usable = props.rows.filter((row) => row.mean_cost_usd);
  const anchor = usable.reduce(
    (best: Json | null, row) => (!best || row.mean_cost_usd < best.mean_cost_usd ? row : best),
    null,
  );
  return (
    <table className="grid">
      <thead>
        <tr>
          <Th help={FUNCTION_MODEL_HELP.model}>model</Th>
          <NumTh help={FUNCTION_MODEL_HELP.calls}>calls</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.runs}>runs</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.args}>args (med)</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.result}>result (med)</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.errors}>errors</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.costPerCall}>$/call</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.observed}>observed ×</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.listPrice}>list price ×</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.behaviour}>behaviour ×</NumTh>
          <NumTh help={FUNCTION_MODEL_HELP.spend}>spend</NumTh>
        </tr>
      </thead>
      <tbody>
        {props.rows.map((row) => (
          <tr key={row.model}>
            <td className="nowrap">
              <i className="swatch" style={{ background: modelColor(row.model) }} />
              {row.model}
            </td>
            <td className="num">{count(row.calls)}</td>
            <td className="num">{count(row.trajectories)}</td>
            <td className="num">{tokens(row.median_arg_tokens)}</td>
            <td className="num">{tokens(row.median_result_tokens)}</td>
            <td className={`num${row.error_rate > 0.2 ? " warn-text" : ""}`}>{pct(row.error_rate, 0)}</td>
            <td className="num">{milli(row.mean_cost_usd)}</td>
            {(() => {
              const observed = anchor && row.mean_cost_usd
                ? row.mean_cost_usd / anchor.mean_cost_usd : null;
              const price = anchor && row.price_uncached && anchor.price_uncached
                ? row.price_uncached / anchor.price_uncached : null;
              const behaviour = observed && price ? observed / price : null;
              return (
                <>
                  <td className="num">{observed ? `${observed.toFixed(2)}×` : "—"}</td>
                  <td className="num muted-cell">{price ? `${price.toFixed(2)}×` : "—"}</td>
                  <td className={`num${behaviour && Math.abs(behaviour - 1) > 0.2 ? " warn-text" : ""}`}>
                    {behaviour ? `${behaviour.toFixed(2)}×` : "—"}
                  </td>
                </>
              );
            })()}
            <td className="num">{usd(row.spend_usd)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Samples(props: {
  tool: string; groupBy: string; groupKey: string;
  onOpenTrajectory: (id: string) => void;
}) {
  const [perModel, setPerModel] = useState(2);
  const { data, error } = useApi<Json>(
    `/api/functions/${encodeURIComponent(props.tool)}/samples` +
    `?group_key=${encodeURIComponent(props.groupKey)}&group_by=${props.groupBy}&per_model=${perModel}`,
  );
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Loading examples…</p>;

  return (
    <Panel
      title={`What each model actually emitted — ${props.tool} · ${props.groupKey}`}
      help="Median-length calls for each model, not the first ones: the opening calls of a run are systematically setup, which would bias every column the same way."
      right={
        <select value={perModel} onChange={(event) => setPerModel(Number(event.target.value))}>
          {[1, 2, 3, 5].map((n) => <option key={n} value={n}>{n} per model</option>)}
        </select>
      }
    >
      <div className="sample-grid">
        {data.models.map((entry: Json) => (
          <div className="sample-model" key={entry.model}>
            <header>
              <span>
                <i className="swatch" style={{ background: modelColor(entry.model) }} />
                {entry.model}
              </span>
              <em>{count(entry.calls)} calls in this group</em>
            </header>
            {entry.samples.map((sample: Json) => (
              <div className="sample" key={sample.fc_id}>
                <div className="sample-meta">
                  <button className="linkish mono" onClick={() => props.onOpenTrajectory(sample.trajectory_id)}>
                    {sample.trajectory_id}
                  </button>
                  <span>call #{sample.call_index}</span>
                  <span>{tokens(sample.arg_tokens)} tok out</span>
                  <span>{tokens(sample.result_tokens)} tok back</span>
                  <strong>{milli(sample.cost_share_usd)}</strong>
                  {sample.result_error && <span className="chip warn small">result reads as error</span>}
                  {sample.n_calls_in_request > 1 && (
                    <span className="chip muted small">1 of {sample.n_calls_in_request} in the request</span>
                  )}
                </div>
                <pre className="raw">{sample.arg_text}</pre>
                {sample.arg_chars > data.sample_chars && (
                  <p className="panel-foot">
                    truncated at {count(data.sample_chars)} of {count(sample.arg_chars)} chars
                  </p>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
      {!data.models.length && <p className="panel-foot">No calls in this group.</p>}
    </Panel>
  );
}
