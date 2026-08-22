/** What every column in the UI actually means.
 *
 *  The tables here are dense and most of the numbers are derived, not logged: the
 *  export ships no `usage`, no outputs and no labels, so a header like `cache` or
 *  `behaviour ×` names something this repo computed rather than something Viktor
 *  recorded. Each string below says which, and where the number can be wrong.
 *
 *  `COLUMN_HELP` is keyed by the raw `calls` index column names (the Raw calls tab
 *  builds its columns straight from the DuckDB schema, so its tooltips have to be
 *  looked up by name). The per-table maps below it are for the curated tables,
 *  whose headers are prose labels rather than column names. */

const ESTIMATED = "Estimated (serialized chars ÷ 4), not measured usage.";

export const COLUMN_HELP: Record<string, string> = {
  call_id: "Primary key of the row: trajectory id and call index joined by a colon. Minted here, not in the export.",
  trajectory_id: "The run this request belongs to — sha1 of the trajectory's opening items, the same id scripts/trajectory_ids.py mints. The export ships no ids of its own.",
  call_index: "Position of this request inside its run, counting from 0. Call 0 is the cold start: it has no cached prefix to hit.",
  n_calls_in_trajectory: "How many requests the agent loop made in this run in total.",
  is_final: "True for the last request of the run. Its output is the one thing the export cannot recover, so output tokens and output cost are null there.",
  model: "The model that actually served this request, as logged. One model serves every call of a trajectory.",
  model_family: "claude, gpt, or other — derived from the model id prefix.",
  tier_rank: "Rank of the model's uncached input price among all models seen, 1 = cheapest. A dense rank: models priced the same share a rank.",
  est_input_tokens: `Prompt items sent with this request, in tokens. Excludes the tool schemas — see tools_tokens for those. ${ESTIMATED}`,
  est_new_tokens: `Tokens this request added on top of the previous request's input — the growth one turn of the loop caused. ${ESTIMATED}`,
  shared_prefix_tokens: "Tokens this request shares as a literal prefix with the previous request's input. This is what a provider could serve from cache — inferred from item-level overlap, because the export has no usage.cached_tokens.",
  prefix_share: "shared_prefix_tokens ÷ est_input_tokens: the fraction of this prompt that was already sent on the previous call. High on a long run, 0 on call 0 and after any model switch.",
  n_items: "How many items the request's `input` array holds — messages, function calls, tool results, reasoning blocks.",
  item_type_sequence: "The types of those items in order, comma-joined. The fingerprint of the loop's shape at this point.",
  item_tokens: `Per-item token estimate for the request's input, in item order. ${ESTIMATED}`,
  last_item_type: "Type of the final input item — what the model is being asked to react to (usually a tool result or a user message).",
  tools_available: "Names of the function definitions handed to this request. They are resent on every call and billed as input.",
  tools_used_so_far: "Distinct functions already called at some point in this run, in first-use order.",
  last_tool_name: "The most recent function named anywhere in the input — normally the tool whose result the model is now reading.",
  n_distinct_tools: "How many different functions this run has used up to this point.",
  has_image: "Whether the input carries an image part. Images are placeholder data URLs in the export, so this flags shape, not content.",
  has_reasoning: "Whether the input carries reasoning items. Only the gpt family emits them, and their summaries are empty in the export.",
  tool_output_error: "The last tool result matched an error pattern (error/failed/exception/traceback/permission denied/…). A keyword match on the result text, not a status code — expect false positives.",
  repeat_tool: "The run has called the same function more than once by this point. A crude retry/loop signal.",
  output_kind: "What the model produced on this call: tool_call, multi_tool, patch, text, or unobservable for the final turn. Recovered from the NEXT request's input, since the export has no outputs.",
  output_n_tool_calls: "How many function calls this one response emitted.",
  output_tool_names: "Which functions this response called.",
  output_prose_tokens: `Tokens of prose in the response, excluding the function-call arguments. ${ESTIMATED}`,
  routine: "Optional label from a router ledger in results/: whether this call was judged routine enough for a cheap model. Blank unless a router wrote it, and always blank on the final call.",
  chosen_model: "The model a router in results/*.jsonl would have used for this call. Blank when no router output joined.",
  kept_logged: "Whether the router's choice equals the model that actually ran.",
  direction: "How the router's choice compares in price to the logged model: upgraded, downgraded, or kept.",
  p_routine: "Optional router confidence that this call is routine, 0–1. Only present if the router wrote it.",
  flip_lambda: "Optional: the cost/quality trade-off weight at which this call's decision flips. Only present if the router wrote it.",
  cost_usd: "What the routed model would have cost for this call, cache-aware: a switch away from the previous call's model resets the cache, so the whole prefix is billed uncached. Blank when no router output joined.",
  switch_penalty_usd: "The extra spend the route's model switch caused at this call — the cacheable prefix re-billed at the uncached rate. 0 when the route stays on one model.",
  split: "Optional train/test split label from a router ledger.",
  source_kind: "Which directory this row came from: export (raw trajectories) or results (router output).",
  source_path: "File this row was read from, relative to the repo root.",
  source_line: "1-based line number in that file.",
  raw_id: "Internal id of the stored raw JSON line this call was reconstructed from.",
  delta_raw_id: "Raw line holding the items this call added to the previous prompt.",
  delta_start_item: "First item index of that delta inside the raw line.",
  delta_stop_item: "Item index just past the end of that delta.",
  output_raw_id: "Raw line in which this call's output is sitting — the export records call k's output inside call k+1's input.",
  output_start_item: "First item index of the recovered output inside that raw line.",
  output_stop_item: "Item index just past the end of the recovered output.",
  ledger_raw_id: "Raw line of the results/*.jsonl record that supplied this row's routing fields.",
  logged_input_cost_usd: "Input cost of this call at the logged model's official rate, cache-aware: prompt + tool schemas, with the shared prefix billed at the cached rate.",
  tools_tokens: `Tokens of the tool schemas sent with this request. They are resent every call and billed as input, and they join the cached prefix from call 1 onward. ${ESTIMATED}`,
  est_output_tokens: `Tokens the model produced, recovered from the next request's input. Null on the final call, whose output the export never captured. ${ESTIMATED}`,
  logged_output_cost_usd: "Output cost of this call at the logged model's official rate. Null on the final call.",
  logged_total_cost_usd: "logged_input_cost_usd + logged_output_cost_usd — what this call cost on the route that actually ran.",
};

export function columnHelp(name: string): string | undefined {
  return COLUMN_HELP[name];
}

/** Trajectories tab — one row per reconstructed run. */
export const TRAJECTORY_HELP = {
  trajectory: "One reconstructed run: every request sharing an opening system prompt and first user message, ordered by input length. The id is a hash of that opening — the export ships no ids. Click the row for its call-by-call flow.",
  model: "The model that served this run. One model serves every call of a trajectory, so this is the whole route; the variation across runs is the natural experiment.",
  calls: "Billed requests the agent loop made inside this run. The ⚠ badge counts calls that opened on a tool result reading like an error.",
  input: `Input tokens billed across every call: prompt items plus the tool schemas, which are resent on each request. ${ESTIMATED}`,
  output: `Output tokens across the run, recovered by reading each call's output out of the next call's input. The final call's output is not in the export, so it is missing from this sum. ${ESTIMATED}`,
  cache: "Share of billed input tokens already present in the previous call's prompt, so a provider would bill them at the cached rate. Inferred from prefix overlap — the export has no usage.cached_tokens.",
  class: "Labels read out of the request itself: workload, size band, and the loop-shape signature (the sequence of moves the run made). Nothing here was guessed by a model.",
  cost: "Input + output cost of the run at the logged model's official rate, cache-aware. The bar is relative to the most expensive run in this list.",
};

/** Function calls tab — the per-function table. */
export const FUNCTION_HELP = {
  function: "The function the model actually called, read out of the function_call items recovered from the following request.",
  calls: "How many times any model emitted a call to this function.",
  models: "Distinct models that called it. Two or more is what makes the row a comparison rather than a description.",
  args: `Median size of the call's arguments. ${ESTIMATED}`,
  result: `Median size of the result that came back — the tokens this tool adds to every later prompt in the run. ${ESTIMATED}`,
  errors: "Share of calls whose result text matched an error pattern. A keyword match, not a status code.",
  costPerCall: "Mean cost of the request that produced the call, in milli-dollars. When one request emitted several calls at once its cost is split evenly across them — an attribution, not a measurement.",
  spend: "Total attributed cost of every call to this function, across all models.",
  spread: "Most expensive model per call ÷ cheapest, for this function. A crude ceiling on what routing this one function could save.",
};

/** Function calls tab — the same function, split by the model that emitted it. */
export const FUNCTION_MODEL_HELP = {
  model: "The model that emitted these calls. The function is held fixed, so differences down the row are differences in how the models use it.",
  calls: "Calls to this function emitted by this model.",
  runs: "Distinct trajectories those calls came from. A small number means one or two runs drive the whole row.",
  args: `Median argument size for this model's calls. ${ESTIMATED}`,
  result: `Median size of the results this model got back. ${ESTIMATED}`,
  errors: "Share of this model's calls whose result text matched an error pattern.",
  costPerCall: "Mean attributed cost per call, in milli-dollars.",
  observed: "This model's $/call ÷ the cheapest-per-call model's in this table. What it actually cost.",
  listPrice: "This model's uncached input rate ÷ the cheapest model's. What the rate card alone predicts.",
  behaviour: "observed × ÷ list price ×. Above 1.0 the model reaches this function with a bigger prompt or writes more when it gets there; below 1.0 it is leaner than its price suggests. Highlighted past ±20%.",
  spend: "Total attributed cost of this model's calls to this function.",
};

/** Compare tab — matched cohorts, the off-policy estimate. */
export const COHORT_HELP = {
  cohort: "The bucket: trajectories sharing the values of the selected matching axes — all things a router could read before the first token.",
  runs: "Trajectories in the bucket, summed over the models that served them.",
  models: "Distinct models that ran inside the bucket. Only buckets with 2+ appear; a single-model bucket has nothing to compare against.",
  actual: "What the runs in this bucket really cost, at the rates of the models that served them.",
  cheapest: "The model with the lowest mean cost per run inside this bucket — the counterfactual route the headroom assumes.",
  headroom: "actual − (cheapest model's mean cost × runs in the bucket). An upper bound: assignment was not random, so part of this gap is the dispatcher having sent harder work to the pricier model.",
  sizeSpread: "Largest ÷ smallest mean billed input tokens across the models in this bucket. Near 1.0 means the models really were seeing the same amount of work; a large value means the headroom is mostly selection, not price.",
};

/** Files tab — export lines. */
export const EXPORT_ROW_HELP = {
  line: "1-based line number in the file. Click the row to see its raw JSON.",
  trajectory: "The trajectory this line was decomposed into. Click to open its call-by-call flow.",
  model: "The model recorded on this line — the one that served every call inside it.",
  items: "Items in the line's `input` array: messages, function calls, tool results, reasoning blocks.",
  calls: "Billed requests scripts/decompose.py unfolded out of this one line.",
  input: `Input tokens billed across those calls, prompt plus tool schemas. ${ESTIMATED}`,
  output: `Output tokens recovered across those calls; the run's final output is not in the export. ${ESTIMATED}`,
  rawSize: "Size of the raw JSON line as stored, in kilobytes.",
  cost: "Cache-aware input + output cost of the calls reconstructed from this line, at the logged model's official rate.",
};

/** Files tab — results (router output) lines. */
export const RESULT_ROW_HELP = {
  line: "1-based line number in the file. Click the row to see its raw JSON.",
  trajectoryId: "The id the router wrote. It only lines up with the export if it was minted the same way — sha1 over the trajectory's opening items.",
  loggedModel: "The model this record says actually served the run.",
  routeLength: "How many per-call decisions the record carries — the length of its `route` array.",
  joins: "Whether that id matches a trajectory reconstructed from export/. No match means every routed-cost column on every tab reads blank.",
  rawSize: "Size of the raw JSON line as stored, in bytes.",
};

/** Overview tab — the source file table. */
export const SOURCE_HELP = {
  kind: "export = raw trajectories, the source of every token count and cost here. results = router output, joined on trajectory id.",
  file: "Path relative to the repo root. These are the only files the explorer reads.",
  lines: "JSONL lines in the file.",
  parseErrors: "Lines that could not be parsed as JSON and were skipped. Anything above 0 means part of the file is missing from every number here.",
  size: "File size on disk.",
};
