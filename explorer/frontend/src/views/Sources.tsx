import { useState } from "react";
import { Json, useApi } from "../api";
import { Note, Panel } from "../charts";
import { count, modelColor, tokens, usd } from "../format";

/** The files on disk, and what each line of them actually became. */
export function Sources(props: { onOpenTrajectory: (id: string) => void }) {
  const { data, error } = useApi<{ files: Json[] }>("/api/sources");
  const [open, setOpen] = useState<string | null>(null);
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Reading sources…</p>;

  return (
    <>
      <Note>
        These are the only files the explorer reads. An <strong>export</strong> line is a
        whole trajectory — the agent loop ran inside it, so one line unfolds into many billed
        requests. A <strong>results</strong> line is a router's decision for one trajectory,
        and only shows up in the cost columns if its trajectory id was minted the same way.
      </Note>

      {data.files.map((file) => {
        const contributed = file.contributed;
        const router = file.router_entries;
        const selected = open === file.path;
        return (
          <Panel
            key={`${file.kind}:${file.path}`}
            title={file.path}
            help={
              file.kind === "export"
                ? "Raw trajectories. Every class, token count and cost on the other tabs is derived from this file."
                : "Router output. Joined to the calls on trajectory id."
            }
            right={
              <button className="ghost" onClick={() => setOpen(selected ? null : file.path)}>
                {selected ? "hide rows" : "browse rows"}
              </button>
            }
          >
            <dl className="kv wide">
              <div><dt>kind</dt><dd><span className={`tag ${file.kind}`}>{file.kind}</span></dd></div>
              <div><dt>lines</dt><dd>{count(file.line_count)}</dd></div>
              <div><dt>on disk</dt><dd>{(file.size_bytes / 1e6).toFixed(1)} MB</dd></div>
              <div>
                <dt>parse errors</dt>
                <dd className={file.parse_errors ? "warn-text" : ""}>{count(file.parse_errors)}</dd>
              </div>
              {contributed && (
                <>
                  <div><dt>calls reconstructed</dt><dd>{count(contributed.calls)}</dd></div>
                  <div><dt>trajectories</dt><dd>{count(contributed.trajectories)}</dd></div>
                  <div><dt>models seen</dt><dd>{count(contributed.models)}</dd></div>
                  <div><dt>est. input / output</dt><dd>{tokens(contributed.billed_input_tokens)} / {tokens(contributed.output_tokens)}</dd></div>
                  <div><dt>est. cost</dt><dd>{usd(contributed.cost_usd)}</dd></div>
                </>
              )}
              {router && (
                <>
                  <div><dt>routing entries</dt><dd>{count(router.entries)}</dd></div>
                  <div><dt>distinct trajectory ids</dt><dd>{count(router.trajectory_ids)}</dd></div>
                </>
              )}
            </dl>
            {(file.error_samples ?? []).length > 0 && (
              <p className="panel-foot warn">
                first parse errors: {(file.error_samples as string[]).join(" · ")}
              </p>
            )}
            {selected && <RowBrowser path={file.path} onOpenTrajectory={props.onOpenTrajectory} />}
          </Panel>
        );
      })}
    </>
  );
}

const PAGE = 100;

function RowBrowser(props: { path: string; onOpenTrajectory: (id: string) => void }) {
  const [offset, setOffset] = useState(0);
  const [line, setLine] = useState<number | null>(null);
  const { data, error } = useApi<Json>(
    `/api/sources/rows?path=${encodeURIComponent(props.path)}&offset=${offset}&limit=${PAGE}`,
  );
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Loading rows…</p>;
  const isExport = data.kind === "export";
  const maxCost = Math.max(...data.rows.map((row: Json) => row.cost_usd ?? 0), 0);

  return (
    <div className="row-browser">
      <div className="row-pager">
        <button className="ghost small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>←</button>
        <span className="range">
          line {count(offset + 1)}–{count(Math.min(offset + PAGE, data.total))} of {count(data.total)}
        </span>
        <button className="ghost small" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>→</button>
      </div>

      <table className="grid rows">
        <thead>
          {isExport ? (
            <tr>
              <th className="num">line</th><th>trajectory</th><th>model</th>
              <th className="num">items</th><th className="num">calls</th>
              <th className="num">input</th><th className="num">output</th>
              <th className="num">raw size</th><th className="num">cost</th>
            </tr>
          ) : (
            <tr>
              <th className="num">line</th><th>trajectory id in this file</th>
              <th>logged model</th><th className="num">route length</th>
              <th>joins to export?</th><th className="num">raw size</th>
            </tr>
          )}
        </thead>
        <tbody>
          {data.rows.map((row: Json) => (
            <tr
              key={row.source_line}
              className={line === row.source_line ? "selected" : ""}
              onClick={() => setLine(line === row.source_line ? null : row.source_line)}
            >
              <td className="num mono">{row.source_line}</td>
              {isExport ? (
                <>
                  <td className="mono">
                    <button
                      className="linkish"
                      onClick={(event) => { event.stopPropagation(); props.onOpenTrajectory(row.trajectory_id); }}
                    >
                      {row.trajectory_id}
                    </button>
                  </td>
                  <td><i className="swatch" style={{ background: modelColor(row.model) }} />{row.model}</td>
                  <td className="num">{count(row.n_items)}</td>
                  <td className="num">{count(row.calls)}</td>
                  <td className="num">{tokens(row.billed_input_tokens)}</td>
                  <td className="num">{tokens(row.output_tokens)}</td>
                  <td className="num">{(row.raw_chars / 1000).toFixed(0)} kB</td>
                  <td className="num cost-cell">
                    <span className="cost-bar" style={{ width: `${maxCost ? ((row.cost_usd ?? 0) / maxCost) * 100 : 0}%` }} />
                    <strong>{usd(row.cost_usd)}</strong>
                  </td>
                </>
              ) : (
                <>
                  <td className="mono">{row.trajectory_id}</td>
                  <td>{row.model}</td>
                  <td className="num">{count(row.calls)}</td>
                  <td>
                    {row.joins_to_export
                      ? <span className="chip route">joined</span>
                      : <span className="chip warn">no match in export</span>}
                  </td>
                  <td className="num">{row.raw_chars} B</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>

      {line !== null && <RawLine path={props.path} line={line} onOpenTrajectory={props.onOpenTrajectory} />}
    </div>
  );
}

function RawLine(props: { path: string; line: number; onOpenTrajectory: (id: string) => void }) {
  const { data, error } = useApi<Json>(
    `/api/sources/row?path=${encodeURIComponent(props.path)}&line=${props.line}`,
  );
  if (error) return <p className="fatal">{error}</p>;
  if (!data) return <p className="loading">Loading line…</p>;
  return (
    <div className="raw-line">
      <div className="raw-line-head">
        <span>
          line {props.line} · {count(data.total_chars)} chars
          {data.shape?.model && <> · {data.shape.model}</>}
          {data.shape?.n_input_items != null && <> · {count(data.shape.n_input_items)} input items</>}
          {data.shape?.tools?.length ? <> · {data.shape.tools.length} tool definitions</> : null}
        </span>
        {data.trajectory_id && (
          <button className="ghost small" onClick={() => props.onOpenTrajectory(data.trajectory_id)}>
            open flow for {data.trajectory_id}
          </button>
        )}
      </div>
      {data.shape?.tools?.length ? (
        <p className="chips">
          {(data.shape.tools as string[]).map((tool) => <span className="chip tool" key={tool}>{tool}</span>)}
        </p>
      ) : null}
      <pre className="raw">{data.preview}</pre>
      {data.truncated && (
        <p className="panel-foot">
          Truncated at {count(data.preview.length)} of {count(data.total_chars)} characters —
          the readable version of this row is its Flow.
        </p>
      )}
    </div>
  );
}
