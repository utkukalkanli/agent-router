import { useState } from "react";
import { TrajectoryDetail, useApi } from "../api";
import { Note, Panel } from "../charts";
import { KIND_COLOR, count, modelColor, pct, shortId, tokens, usd } from "../format";

const ITEM_LABEL: Record<string, string> = {
  sys: "system", user: "user", assistant: "assistant",
  fc: "tool call", fco: "tool result", ctc: "custom tool call",
  ctco: "custom tool result", reasoning: "reasoning",
  function_call: "tool call", function_call_output: "tool result",
  custom_tool_call: "custom tool call", custom_tool_call_output: "custom tool result",
  message: "message",
};

/** One trajectory, call by call: what each request paid for and what came back. */
export function Flow(props: {
  trajectoryId: string;
  onCompare: (id: string) => void;
  onClose: () => void;
}) {
  const { data, error } = useApi<TrajectoryDetail>(`/api/trajectories/${props.trajectoryId}`);
  const [openCall, setOpenCall] = useState<number | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Loading trajectory…</p>;

  const t = data.trajectory;
  const maxBilled = Math.max(
    ...data.calls.map((call) => (call.est_input_tokens ?? 0) + (call.tools_tokens ?? 0) + (call.est_output_tokens ?? 0)),
    1,
  );
  const total = t.cost_usd ?? 0;

  return (
    <div className="flow">
      <header className="flow-head">
        <div>
          <h2>
            <span className="mono">{t.trajectory_id}</span>
            <i className="swatch" style={{ background: modelColor(t.model) }} />
            {t.model}
          </h2>
          <p className="flow-meta">
            {count(t.n_calls)} calls · {tokens(t.billed_input_tokens)} est. input tokens ·
            {" "}{tokens(t.total_est_output_tokens)} recovered output · cache {pct(t.cache_hit_rate, 0)}
            {t.started_at && <> · started {t.started_at}</>}
          </p>
          <p className="chips">
            <span className="chip">{t.workload}</span>
            <span className="chip">{t.trigger_kind}</span>
            <span className="chip">{t.harness}</span>
            <span className="chip">{t.surface}</span>
            <span className="chip">{t.size_band}</span>
            <span className="chip">{t.depth_band}</span>
            <span className="chip mono">{t.shape_signature}</span>
            {t.cron_family && (
              <span className={`chip${t.cron_family_collapsed ? " warn" : ""}`}>
                cron:{t.cron_family}{t.cron_family_collapsed ? " (redaction-collapsed)" : ""}
              </span>
            )}
            {t.has_error && <span className="chip warn">{count(t.n_error_calls)} calls saw a tool error</span>}
          </p>
        </div>
        <div className="flow-actions">
          <strong className="flow-cost">{usd(total)}</strong>
          <span className="flow-cost-note">
            {usd(t.input_cost_usd)} in + {usd(t.output_cost_usd)} out
          </span>
          <button className="ghost" onClick={() => props.onCompare(t.trajectory_id)}>compare with…</button>
          <button className="ghost" onClick={props.onClose}>close</button>
        </div>
      </header>

      <div className="flow-legend">
        <span><i style={{ background: "var(--c-cached)" }} />cached prefix</span>
        <span><i style={{ background: "var(--c-new)" }} />new input since last call</span>
        <span><i style={{ background: "var(--c-out)" }} />recovered output</span>
        <span className="flow-legend-note">bar width = share of the largest call in this run</span>
      </div>

      <ol className="calls">
        {data.calls.map((call) => {
          const cached = call.shared_prefix_tokens ?? 0;
          const fresh = Math.max((call.est_input_tokens ?? 0) - cached, 0) + (call.tools_tokens ?? 0);
          const out = call.est_output_tokens ?? 0;
          const billed = cached + fresh + out;
          const open = openCall === call.call_index;
          return (
            <li key={call.call_id} className={`call${open ? " open" : ""}${call.tool_output_error ? " has-error" : ""}`}>
              <button className="call-head" onClick={() => setOpenCall(open ? null : call.call_index)}>
                <span className="call-index">{call.call_index}</span>
                <span className="call-body">
                  <span className="call-line">
                    <span
                      className="kind-dot"
                      style={{ background: KIND_COLOR[call.output_kind] ?? "var(--muted)" }}
                      title={call.output_kind}
                    />
                    <span className="call-kind">{call.output_kind}</span>
                    {(call.output_tool_names ?? []).map((name: string, index: number) => (
                      <span className="chip tool" key={`${name}-${index}`}>{name}</span>
                    ))}
                    {call.is_final && <span className="chip muted">final turn — output not in the export</span>}
                    {call.tool_output_error && <span className="chip warn">previous result looks like an error</span>}
                  </span>
                  <span className="call-bar" style={{ width: `${(billed / maxBilled) * 100}%` }}>
                    <i style={{ flexGrow: cached, background: "var(--c-cached)" }} title={`${count(cached)} cached tokens`} />
                    <i style={{ flexGrow: fresh, background: "var(--c-new)" }} title={`${count(fresh)} new input tokens`} />
                    <i style={{ flexGrow: out, background: "var(--c-out)" }} title={`${count(out)} output tokens`} />
                  </span>
                  <span className="call-numbers">
                    {tokens(call.est_input_tokens)} in ({pct(call.prefix_share, 0)} cached) ·
                    {" "}{tokens(call.est_output_tokens)} out
                    {call.direction && call.direction !== "kept" && (
                      <span className="chip route">{call.direction} → {call.chosen_model}</span>
                    )}
                  </span>
                </span>
                <span className="call-cost">
                  <strong>{usd(call.logged_total_cost_usd)}</strong>
                  <em>{pct(total ? (call.logged_total_cost_usd ?? 0) / total : null, 0)}</em>
                </span>
              </button>
              {open && (
                <div className="call-detail">
                  <dl className="kv">
                    <div><dt>prompt items</dt><dd>{count(call.n_items)}</dd></div>
                    <div><dt>tool schemas</dt><dd>{tokens(call.tools_tokens)} tok, resent every call</dd></div>
                    <div><dt>new since last call</dt><dd>{tokens(call.est_new_tokens)} tok</dd></div>
                    <div><dt>input / output cost</dt><dd>{usd(call.logged_input_cost_usd)} / {usd(call.logged_output_cost_usd)}</dd></div>
                    {call.chosen_model && (
                      <div><dt>router picked</dt><dd>{call.chosen_model} → {usd(call.routed_cost_usd)}</dd></div>
                    )}
                    {call.switch_penalty_usd > 0 && (
                      <div><dt>cache reset penalty</dt><dd className="warn-text">{usd(call.switch_penalty_usd)}</dd></div>
                    )}
                  </dl>
                  <ItemStream items={data.items.filter((item) => item.call_index === call.call_index)} />
                </div>
              )}
            </li>
          );
        })}
      </ol>

      <Panel
        title="Opening request"
        help="The first user message. Viktor packs the thread info, cron memory and channel instructions into it — that is where every class on this run comes from."
        right={<button className="ghost" onClick={() => setShowPrompt(!showPrompt)}>{showPrompt ? "hide" : "show"}</button>}
      >
        {showPrompt ? (
          <pre className="raw">{data.opening_user ?? "(none)"}</pre>
        ) : (
          <p className="panel-foot">
            {(data.opening_user ?? "").length.toLocaleString()} characters ·
            {t.cron_path && <> cron path <code>{t.cron_path}</code> ·</>}
            {t.thread_path && <> thread <code>{t.thread_path}</code></>}
          </p>
        )}
      </Panel>

      <Note>
        Source: <code>{t.source_path}</code> line {t.source_line} · trajectory id{" "}
        <code>{shortId(t.trajectory_id)}</code> is the SHA-1 of the item list, the same id{" "}
        <code>scripts/decompose.py</code> and <code>results/routes.jsonl</code> use.
      </Note>
    </div>
  );
}

function ItemStream(props: { items: any[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  if (!props.items.length) return null;
  return (
    <ul className="items">
      {props.items.map((item) => (
        <li key={item.index} className={item.is_output ? "model" : "env"}>
          <button onClick={() => setExpanded(expanded === item.index ? null : item.index)}>
            <span className={`item-type ${item.is_output ? "model" : "env"}`}>
              {ITEM_LABEL[item.type] ?? item.type}
            </span>
            {item.tool && <span className="chip tool">{item.tool}</span>}
            <span className="item-tokens">{tokens(item.tokens)} tok</span>
            <span className="item-peek">{(item.text ?? "").slice(0, 130).replace(/\s+/g, " ")}</span>
          </button>
          {expanded === item.index && <pre className="raw">{item.text}</pre>}
        </li>
      ))}
    </ul>
  );
}
