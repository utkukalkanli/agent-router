const COMPACT = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const PLAIN = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function tokens(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  return Number.isFinite(n) ? COMPACT.format(n) : "—";
}

export function count(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  return Number.isFinite(n) ? PLAIN.format(n) : "—";
}

/** Costs span six orders of magnitude here, so scale the precision to the value. */
export function usd(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const magnitude = Math.abs(n);
  if (magnitude >= 100) return `$${n.toFixed(0)}`;
  if (magnitude >= 1) return `$${n.toFixed(2)}`;
  if (magnitude >= 0.01) return `$${n.toFixed(3)}`;
  return `$${n.toFixed(5)}`;
}

export function signedUsd(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n < 0 ? "−" : "+"}${usd(Math.abs(n))}`;
}

export function pct(value: unknown, digits = 1): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "—";
}

export function signedPct(value: unknown, digits = 1): string {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n >= 0 ? "+" : ""}${(n * 100).toFixed(digits)}%`;
}

export function shortId(value: string | null | undefined): string {
  return value ? value.slice(0, 8) : "—";
}

/** Stable per-model colour so a model reads the same in every chart on every tab. */
const MODEL_HUES: Record<string, number> = {
  "claude-fable-5": 292, "claude-opus-5": 258, "claude-opus-4-8": 246,
  "claude-opus-4-6": 234, "claude-sonnet-5": 199, "claude-sonnet-4-6": 186,
  "gpt-5.6-sol": 28, "gpt-5.6-terra": 46, "gpt-5.6-luna": 160,
};

export function modelColor(model: string | null | undefined): string {
  if (!model) return "var(--muted)";
  const hue = MODEL_HUES[model] ?? ([...model].reduce((a, c) => a + c.charCodeAt(0), 0) % 360);
  return `hsl(${hue} 62% 58%)`;
}

export const KIND_COLOR: Record<string, string> = {
  tool_call: "var(--c-tool)", multi_tool: "var(--c-multi)", patch: "var(--c-patch)",
  text: "var(--c-text)", unobservable: "var(--c-unobs)", unknown: "var(--muted)",
};

export const WORK_COLOR: Record<string, string> = {
  exec: "var(--c-tool)", edit: "var(--c-patch)", read: "var(--c-text)",
  message: "var(--c-multi)", wait: "var(--c-unobs)", other: "var(--muted)",
  none: "var(--muted)",
};
