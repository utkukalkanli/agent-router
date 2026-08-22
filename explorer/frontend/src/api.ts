export type Json = Record<string, any>;

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return (await response.json()) as T;
}

/** Fetch on mount / whenever `path` changes, with abort on unmount. */
import { useEffect, useState } from "react";

export function useApi<T>(path: string | null): {
  data: T | null; error: string | null; loading: boolean;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!path) { setData(null); return; }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    get<T>(path, controller.signal)
      .then(setData)
      .catch((reason) => { if (reason.name !== "AbortError") setError(String(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [path]);

  return { data, error, loading };
}

export type Overview = {
  headline: Json;
  by_model: Json[];
  by_call_index: Json[];
  concentration: Json[];
  by_output_kind: Json[];
  sources: Json[];
  route_join: Json | null;
};
export type ClassAxis = {
  key: string; label: string; help: string; limit?: number; rows: Json[];
};
export type TrajectoryRow = Json & { trajectory_id: string };
export type TrajectoryList = {
  rows: TrajectoryRow[]; summary: Json;
  facets: Record<string, Json[]>;
  sort: string; order: string; limit: number; offset: number;
};
export type TrajectoryDetail = {
  trajectory: TrajectoryRow; calls: Json[]; items: Json[];
  opening_user: string | null; system_prompt: string | null;
};
export type CompareResult = {
  left: { trajectory: TrajectoryRow; steps: string[] };
  right: { trajectory: TrajectoryRow; steps: string[] };
  steps: Json[]; summary: Json;
};
export type Matched = { axes: string[]; cohorts: Json[]; coverage: Json };
export type ToolProfile = {
  tools: Json[]; by_work: Json[]; total_added_prompt_tokens: number;
};
