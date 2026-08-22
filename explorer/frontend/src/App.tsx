import {
  ColumnDef,
  SortingState,
  VisibilityState,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  KeyboardEvent as ReactKeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type CallRow = Record<string, unknown> & { call_id: string };
type ColumnInfo = { name: string; type: string; nullable: boolean };
type SourceInfo = {
  kind: string; path: string; line_count: number; parse_errors: number;
  size_bytes: number; mtime_ns: number;
};
type SchemaResponse = {
  columns: ColumnInfo[]; default_columns: string[]; sources: SourceInfo[];
  source_kinds: string[]; calls: number; trajectories: number;
  token_estimator: string | null;
};
type CallsResponse = {
  rows: CallRow[]; total: number; offset: number; limit: number;
  sort: string; order: "asc" | "desc"; elapsed_ms: number;
};

const PAGE_SIZE = 200;
const UI_DEFAULTS = new Set([
  "trajectory_id", "call_index", "model", "est_input_tokens",
  "est_new_tokens", "prefix_share", "last_item_type", "last_tool_name",
  "tool_output_error", "repeat_tool", "output_kind", "routine",
  "chosen_model", "direction", "cost_usd",
]);

function label(name: string) {
  return name.replaceAll("_", " ");
}

function formatValue(value: unknown, column: string) {
  if (value === null || value === undefined) return <span className="null">null</span>;
  if (typeof value === "boolean") return <span className={value ? "yes" : "no"}>{value ? "yes" : "no"}</span>;
  if (Array.isArray(value)) return value.length ? value.join(", ") : <span className="empty">[]</span>;
  if (typeof value === "number") {
    if (column.includes("cost") || column.includes("penalty")) return value.toFixed(6);
    if (column.includes("share") || column.startsWith("p_") || column === "flip_lambda") return value.toFixed(3);
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
  }
  return String(value);
}

function useCalls(schema: SchemaResponse | null, visible: string[], sorting: SortingState, offset: number) {
  const [result, setResult] = useState<CallsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const selectedColumns = useMemo(() => {
    const names = schema?.columns.map((column) => column.name) ?? [];
    return ["call_id", ...visible].filter((value, index, all) => names.includes(value) && all.indexOf(value) === index);
  }, [schema, visible]);

  useEffect(() => {
    if (!schema || !selectedColumns.length) return;
    const controller = new AbortController();
    const currentSort = sorting[0] ?? { id: "call_id", desc: false };
    const params = new URLSearchParams({
      offset: String(offset), limit: String(PAGE_SIZE), sort: currentSort.id,
      order: currentSort.desc ? "desc" : "asc", columns: selectedColumns.join(","),
    });
    setLoading(true);
    setError(null);
    fetch(`/api/calls?${params}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return response.json() as Promise<CallsResponse>;
      })
      .then(setResult)
      .catch((reason) => {
        if (reason.name !== "AbortError") setError(String(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [schema, selectedColumns, sorting, offset]);
  return { result, error, loading };
}

export function App() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [sorting, setSorting] = useState<SortingState>([{ id: "call_id", desc: false }]);
  const [offset, setOffset] = useState(0);
  const [columnOrder, setColumnOrder] = useState<string[]>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [columnQuery, setColumnQuery] = useState("");
  const tableScrollRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const draggedColumn = useRef<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/schema", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return response.json() as Promise<SchemaResponse>;
      })
      .then((next) => {
        setSchema(next);
        const names = next.columns.map((column) => column.name);
        setColumnOrder(names);
        setColumnVisibility(Object.fromEntries(names.map((name) => [name, UI_DEFAULTS.has(name)])));
      })
      .catch((reason) => {
        if (reason.name !== "AbortError") setSchemaError(String(reason));
      });
    return () => controller.abort();
  }, []);

  const visibleColumns = useMemo(
    () => columnOrder.filter((name) => columnVisibility[name] !== false),
    [columnOrder, columnVisibility],
  );
  const { result, error, loading } = useCalls(schema, visibleColumns, sorting, offset);
  const rows = result?.rows ?? [];

  const columns = useMemo<ColumnDef<CallRow>[]>(() => (schema?.columns ?? []).map((column) => ({
    id: column.name,
    accessorKey: column.name,
    header: label(column.name),
    size: column.name === "trajectory_id" ? 150 : column.name === "model" ? 145 : 125,
    minSize: 70,
    maxSize: 520,
    cell: (context) => formatValue(context.getValue(), column.name),
  })), [schema]);

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, columnOrder, columnVisibility },
    onSortingChange: (update) => {
      setSorting(update);
      setOffset(0);
    },
    onColumnOrderChange: setColumnOrder,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
    columnResizeMode: "onChange",
    enableMultiSort: false,
  });

  const virtualizer = useVirtualizer({
    count: table.getRowModel().rows.length,
    getScrollElement: () => tableScrollRef.current,
    estimateSize: () => 31,
    overscan: 12,
  });
  const selected = rows[selectedIndex] ?? null;

  useEffect(() => {
    setSelectedIndex(0);
    setDrawerOpen(false);
  }, [offset, sorting]);

  useEffect(() => {
    const listener = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
      if (event.key === "/" && !typing) {
        event.preventDefault();
        searchRef.current?.focus();
      } else if (event.key === "Escape") {
        setDrawerOpen(false);
      } else if (!typing && (event.key === "j" || event.key === "k")) {
        event.preventDefault();
        setSelectedIndex((current) => {
          const next = Math.max(0, Math.min(rows.length - 1, current + (event.key === "j" ? 1 : -1)));
          virtualizer.scrollToIndex(next, { align: "auto" });
          return next;
        });
      } else if (!typing && event.key === "Enter" && selected) {
        setDrawerOpen(true);
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [rows.length, selected, virtualizer]);

  function reorderColumn(target: string) {
    const source = draggedColumn.current;
    if (!source || source === target) return;
    setColumnOrder((current) => {
      const next = current.filter((name) => name !== source);
      next.splice(next.indexOf(target), 0, source);
      return next;
    });
  }

  if (schemaError) return <main className="fatal">Could not load index schema: {schemaError}</main>;
  if (!schema) return <main className="fatal">Loading index schema…</main>;

  const filteredColumnOptions = schema.columns.filter((column) =>
    column.name.toLowerCase().includes(columnQuery.toLowerCase()),
  );
  const firstRow = offset + 1;
  const lastRow = Math.min(offset + rows.length, result?.total ?? 0);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Trajectory Explorer</h1>
          <span className="subtitle">Calls · estimated tokens · local only</span>
        </div>
        <div className="source-status" title="Loaded JSONL sources">
          {schema.source_kinds.map((kind) => {
            const files = schema.sources.filter((source) => source.kind === kind);
            const lines = files.reduce((sum, source) => sum + source.line_count, 0);
            return <span className={`source ${kind}`} key={kind}>{kind} {files.length}f / {lines.toLocaleString()}l</span>;
          })}
          {!schema.sources.length && <span className="source missing">no sources</span>}
        </div>
      </header>

      <nav className="tabs"><button className="active">Calls</button></nav>

      <section className="toolbar">
        <div className="metrics">
          <strong>{schema.calls.toLocaleString()}</strong> calls
          <span>·</span><strong>{schema.trajectories.toLocaleString()}</strong> trajectories
          {result && <span>· query {result.elapsed_ms.toFixed(1)} ms</span>}
        </div>
        <input ref={searchRef} className="search-stub" placeholder="Search (filters checkpoint)" disabled />
        <details className="column-menu">
          <summary>Columns {visibleColumns.length}/{schema.columns.length}</summary>
          <div className="column-popover">
            <input value={columnQuery} onChange={(event) => setColumnQuery(event.target.value)} placeholder="Find column" />
            <div className="column-options">
              {filteredColumnOptions.map((column) => (
                <label key={column.name}>
                  <input
                    type="checkbox"
                    checked={columnVisibility[column.name] !== false}
                    onChange={(event) => setColumnVisibility((current) => ({ ...current, [column.name]: event.target.checked }))}
                  />
                  <span>{label(column.name)}</span><small>{column.type}</small>
                </label>
              ))}
            </div>
          </div>
        </details>
        <button className="pager" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>←</button>
        <span className="range">{result?.total ? `${firstRow.toLocaleString()}–${lastRow.toLocaleString()}` : "0"}</span>
        <button className="pager" disabled={!result || offset + rows.length >= result.total} onClick={() => setOffset(offset + PAGE_SIZE)}>→</button>
      </section>

      {(error || loading) && <div className={error ? "query-state error" : "query-state"}>{error ?? "Querying DuckDB…"}</div>}

      <div className="table-wrap" ref={tableScrollRef} tabIndex={0} onKeyDown={(_event: ReactKeyboardEvent) => undefined}>
        <div className="table-canvas" style={{ width: table.getTotalSize(), height: virtualizer.getTotalSize() + 32 }}>
          <div className="header-row" style={{ width: table.getTotalSize() }}>
            {table.getFlatHeaders().map((header) => (
              <div
                className="header-cell"
                key={header.id}
                style={{ width: header.getSize() }}
                draggable
                onDragStart={() => { draggedColumn.current = header.column.id; }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={() => reorderColumn(header.column.id)}
                onClick={header.column.getToggleSortingHandler()}
                title="Click to sort; drag to reorder"
              >
                <span>{flexRender(header.column.columnDef.header, header.getContext())}</span>
                <i>{header.column.getIsSorted() === "asc" ? "▲" : header.column.getIsSorted() === "desc" ? "▼" : ""}</i>
                <b
                  className="resizer"
                  onMouseDown={(event) => { event.stopPropagation(); header.getResizeHandler()(event); }}
                  onTouchStart={header.getResizeHandler()}
                />
              </div>
            ))}
          </div>
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const row = table.getRowModel().rows[virtualRow.index];
            return (
              <div
                className={`data-row ${selectedIndex === virtualRow.index ? "selected" : ""}`}
                key={row.id}
                style={{ width: table.getTotalSize(), transform: `translateY(${virtualRow.start + 32}px)` }}
                onClick={() => { setSelectedIndex(virtualRow.index); setDrawerOpen(true); }}
              >
                {row.getVisibleCells().map((cell) => (
                  <div className="data-cell" key={cell.id} style={{ width: cell.column.getSize() }} title={String(cell.getValue() ?? "null")}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      <footer>
        <span>{schema.token_estimator}; not measured usage</span>
        <span>/ search · j/k move · enter inspect · esc close</span>
      </footer>

      {drawerOpen && selected && (
        <aside className="drawer" aria-label="Call details">
          <div className="drawer-head">
            <div><strong>{selected.call_id}</strong><span>indexed call fields</span></div>
            <button onClick={() => setDrawerOpen(false)}>×</button>
          </div>
          <dl>
            {Object.entries(selected).map(([key, value]) => (
              <div key={key}><dt>{label(key)}</dt><dd>{formatValue(value, key)}</dd></div>
            ))}
          </dl>
          <p className="drawer-note">Raw source and delta views arrive in their dedicated checkpoints.</p>
        </aside>
      )}
    </div>
  );
}
