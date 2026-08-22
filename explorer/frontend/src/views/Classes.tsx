import { useState } from "react";
import { ClassAxis, useApi } from "../api";
import { BarRow, Note, Panel } from "../charts";
import { count, modelColor, pct, signedPct, usd } from "../format";

/** Axes a trajectory list can be filtered on directly. `cost_usd` here is always the
 *  logged route; `routed` is what results/routes.jsonl would have paid instead. */
export function Classes(props: { onPickClass: (axis: string, value: string) => void }) {
  const { data, error } = useApi<{ axes: ClassAxis[] }>("/api/classes");
  const [open, setOpen] = useState<string>("workload");
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Classifying…</p>;

  return (
    <>
      <Note>
        The export ships no labels. Every class below is read out of the request itself —
        the tool definitions it was sent, the <code>Triggered by:</code> and{" "}
        <code>Cron path:</code> lines Viktor packs into the opening message, and the
        sequence of moves the agent loop actually made. Nothing here is a guess by a model.
      </Note>

      {data.axes.map((axis) => {
        const expanded = open === axis.key;
        const max = Math.max(...axis.rows.map((row) => row.cost_usd ?? 0));
        const total = axis.rows.reduce((sum, row) => sum + (row.cost_usd ?? 0), 0);
        const shown = expanded ? axis.rows : axis.rows.slice(0, 5);
        return (
          <Panel
            key={axis.key}
            title={axis.label}
            help={axis.help}
            right={
              <button className="ghost" onClick={() => setOpen(expanded ? "" : axis.key)}>
                {expanded ? "collapse" : `${axis.rows.length} groups`}
              </button>
            }
          >
            {shown.map((row) => (
              <BarRow
                key={row.value}
                label={
                  <>
                    {row.value}
                    {axis.key === "model" && <i className="swatch" style={{ background: modelColor(row.value) }} />}
                  </>
                }
                sub={
                  <>
                    {count(row.trajectories)} runs · median {count(row.median_calls)} calls ·
                    {" "}cache {pct(row.cache_hit_rate, 0)} ·
                    {" "}{count(row.distinct_models)} model{row.distinct_models === 1 ? "" : "s"}
                    {row.error_trajectories > 0 && <> · {count(row.error_trajectories)} with tool errors</>}
                  </>
                }
                value={row.cost_usd ?? 0}
                max={max}
                text={`${usd(row.cost_usd)} · ${pct((row.cost_usd ?? 0) / total, 0)}`}
                onClick={() => props.onPickClass(axis.key, row.value)}
              />
            ))}
            {!expanded && axis.rows.length > 5 && (
              <p className="panel-foot">
                {axis.rows.length - 5} more groups · click a row to filter the trajectory list
              </p>
            )}
            {axis.key === "model" && (
              <p className="panel-foot warn">
                Reading a cost gap between two rows here as a routing opportunity assumes the
                models were handed comparable work. They were not assigned at random — see
                the Cohorts panel on the Compare tab for the matched version of this question.
              </p>
            )}
            {axis.key === "cron_family" && (
              <p className="panel-foot warn">
                Redaction placeholders are only consistent <em>within</em> a trajectory, so any
                family that is itself a <code>PII_*</code> token silently merges unrelated crons.
                Do not treat those rows as one workload.
              </p>
            )}
          </Panel>
        );
      })}
    </>
  );
}

export function Tools() {
  const { data, error } = useApi<{ tools: any[]; by_work: any[]; total_added_prompt_tokens: number }>("/api/tools");
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Profiling tools…</p>;
  const maxAdded = Math.max(...data.tools.map((tool) => tool.added_prompt_tokens ?? 0));
  const maxWork = Math.max(...data.by_work.map((work) => work.added_prompt_tokens ?? 0));

  return (
    <>
      <Note>
        A tool's real cost is not the call that requests it — it is the tokens its{" "}
        <strong>result</strong> adds to every later prompt in the run. Each row below
        attributes prompt growth to the tool whose output caused it, so this is a
        ranking of which function calls make the trajectory expensive to continue.
      </Note>

      <Panel
        title="Prompt growth by tool"
        help="Tokens this tool's output added to the next prompt, summed over every time it ran. `calls after` is how many requests followed one of its results."
      >
        {data.tools.map((tool) => (
          <BarRow
            key={tool.tool}
            label={<>{tool.tool}<span className={`tag work-${tool.work}`}>{tool.work}</span></>}
            sub={
              <>
                {count(tool.times_issued)} calls issued · median {count(tool.median_added_tokens)} tokens back ·
                {" "}max {count(tool.max_added_tokens)}
                {tool.error_rate > 0.02 && <> · <span className="warn-text">{pct(tool.error_rate, 0)} of results look like errors</span></>}
              </>
            }
            value={tool.added_prompt_tokens ?? 0}
            max={maxAdded}
            color={`var(--c-work-${tool.work})`}
            text={`${count(tool.added_prompt_tokens)} tok · ${pct(tool.added_share, 0)}`}
          />
        ))}
      </Panel>

      <Panel
        title="By kind of work"
        help="The same numbers rolled up into what the model was doing with the turn."
      >
        {data.by_work.map((work) => (
          <BarRow
            key={work.work}
            label={work.work}
            sub={<>{count(work.tools)} tools · {count(work.calls_after)} follow-on calls</>}
            value={work.added_prompt_tokens}
            max={maxWork}
            color={`var(--c-work-${work.work})`}
            text={`${count(work.added_prompt_tokens)} tok · ${usd(work.cost_usd)}`}
          />
        ))}
        <p className="panel-foot">
          Total prompt growth attributed: {count(data.total_added_prompt_tokens)} estimated tokens.
        </p>
      </Panel>
    </>
  );
}
