import { ReactNode } from "react";

/** A labelled horizontal bar in a ranked list. The bar is the comparison; the
 *  number is the receipt. */
export function BarRow(props: {
  label: ReactNode; value: number; max: number; text: string;
  color?: string; sub?: ReactNode; onClick?: () => void; active?: boolean;
}) {
  const width = props.max > 0 ? Math.max((props.value / props.max) * 100, 0.4) : 0;
  return (
    <div
      className={`bar-row${props.onClick ? " clickable" : ""}${props.active ? " active" : ""}`}
      onClick={props.onClick}
      role={props.onClick ? "button" : undefined}
      tabIndex={props.onClick ? 0 : undefined}
      onKeyDown={(event) => { if (props.onClick && event.key === "Enter") props.onClick(); }}
    >
      <div className="bar-label">
        <span className="bar-name">{props.label}</span>
        {props.sub && <span className="bar-sub">{props.sub}</span>}
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${width}%`, background: props.color ?? "var(--accent)" }} />
      </div>
      <span className="bar-value">{props.text}</span>
    </div>
  );
}

export type Segment = { key: string; value: number; color: string; label?: string };

/** One bar split into parts — used for token mix (cached / new / output). */
export function StackedBar(props: { segments: Segment[]; height?: number }) {
  const total = props.segments.reduce((sum, part) => sum + Math.max(part.value, 0), 0);
  if (!total) return <div className="stacked empty" />;
  return (
    <div className="stacked" style={{ height: props.height ?? 14 }}>
      {props.segments.map((part) => (
        <div
          key={part.key}
          className="stacked-part"
          style={{ width: `${(Math.max(part.value, 0) / total) * 100}%`, background: part.color }}
          title={`${part.label ?? part.key}: ${part.value.toLocaleString()}`}
        />
      ))}
    </div>
  );
}

/** Vertical columns with an optional cumulative overlay line. */
export function ColumnChart(props: {
  data: { key: string; value: number; color?: string; title?: string }[];
  height?: number; cumulative?: boolean; format?: (value: number) => string;
}) {
  const height = props.height ?? 150;
  const max = Math.max(...props.data.map((d) => d.value), 0);
  const total = props.data.reduce((sum, d) => sum + d.value, 0);
  let running = 0;
  const points = props.data.map((d, index) => {
    running += d.value;
    const x = ((index + 0.5) / props.data.length) * 100;
    const y = total ? (1 - running / total) * 100 : 100;
    return `${x},${y}`;
  });
  return (
    <div className="column-chart" style={{ height }}>
      <div className="columns">
        {props.data.map((d) => (
          <div className="column" key={d.key} title={d.title ?? `${d.key}: ${props.format?.(d.value) ?? d.value}`}>
            <div
              className="column-fill"
              style={{
                height: `${max ? Math.max((d.value / max) * 100, 0.6) : 0}%`,
                background: d.color ?? "var(--accent)",
              }}
            />
            <span className="column-key">{d.key}</span>
          </div>
        ))}
      </div>
      {props.cumulative && (
        <svg className="cumulative" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
          <polyline points={points.join(" ")} />
        </svg>
      )}
    </div>
  );
}

/** Headline number with its unit and an explanation of what it is measuring. */
export function Stat(props: { label: string; value: string; note?: ReactNode; tone?: "good" | "warn" }) {
  return (
    <div className={`stat${props.tone ? ` ${props.tone}` : ""}`}>
      <span className="stat-label">{props.label}</span>
      <strong className="stat-value">{props.value}</strong>
      {props.note && <span className="stat-note">{props.note}</span>}
    </div>
  );
}

export function Note(props: { children: ReactNode; tone?: "warn" }) {
  return <p className={`note${props.tone ? ` ${props.tone}` : ""}`}>{props.children}</p>;
}

export function Panel(props: { title: string; help?: ReactNode; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <div>
          <h2>{props.title}</h2>
          {props.help && <p className="panel-help">{props.help}</p>}
        </div>
        {props.right}
      </header>
      {props.children}
    </section>
  );
}

/** A table header that explains itself. Almost nothing in these tables is a number
 *  the export logged — most are reconstructed or estimated — so every column gets a
 *  hover/focus note saying what it is and where it can be wrong. */
export function Th(props: { children: ReactNode; help: string; className?: string; align?: "right" }) {
  const className = [props.className, props.align === "right" ? "tip-right" : null]
    .filter(Boolean)
    .join(" ");
  return (
    <th className={className || undefined}>
      <span className="th-help" tabIndex={0}>
        {props.children}
        <span className="th-tip" role="tooltip">{props.help}</span>
      </span>
    </th>
  );
}

/** The same, for the numeric columns: right-aligned, so the note hangs right. */
export function NumTh(props: { children: ReactNode; help: string }) {
  return <Th className="num" align="right" help={props.help}>{props.children}</Th>;
}
