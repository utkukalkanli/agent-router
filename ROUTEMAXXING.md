# Routemaxxing

Routemaxxing minimizes estimated, cache-aware model cost subject to a two-percentage-
point upper bound on automatic failure-proxy degradation. It runs fully offline and
uses only information available before each routing decision.

## The structural finding

The public chunk contains 1,000 unique histories. Their item sequences unfold into
10,845 model turns: model-authored runs produce tool calls or prose, and environment
items close one call and begin the next. The briefing-compatible analysis treats the
last output of every trajectory as unobservable.

The two model families also have disjoint tool dialects. Claude histories expose
`bash`/`file_edit`; GPT histories expose `shell_command`/`apply_patch`. Routemaxxing
therefore routes only within a family rather than inventing an unsupported cross-family
counterfactual.

## Policy

The certified policy reads only call-zero information: the opening task, trigger,
toolset, initial size, images, and nameable intent flags. It generates only strictly
cheaper, propensity-supported downgrades within the tool-dialect family. Proposals must
save money under both fixed-shape and behavior-adjusted cost, are ranked by training-only
partially pooled dollars saved per unit of predicted proxy-failure risk, and are admitted
by a threshold selected on training data. The savings predictor sees opening intent,
initial size, trigger, image, and surface—not realized depth or future outputs. The
selected model is kept for the whole trajectory.

The winning public policy admits every positive-savings supported proposal. This is not
an unbounded extrapolation: the full-adoption point won the training-only sweep while
its routed effective sample size stayed above 15 and its one-sided proxy-failure upper
bound stayed below +2 points. Unsupported defaults fall back unchanged. In particular,
GPT Luna has only 12 training trajectories versus the fixed minimum of 20, so the
certified router never selects it.

The adaptive policy starts from the certified choice and may escalate once after an
observed tool error, repeated action, or high-risk context growth. It never downgrades
or exceeds the original default, and it switches only if savings banked so far pay for
the full cache reset at that call. Its result is explicitly conditional on logged
states: a different early model could have generated a different later context.

## Evaluation

- Deterministic 70/30 split grouped by recoverable thread identity, falling back to the
  stable trajectory hash.
- L2 logistic outcome models and multinomial propensity models trained on the public
  training split. Enriched opening features and partial propensity pooling are reported
  as ablations; neither unlocked more routes than the simpler v1 feature model.
- Downgrade-from-default effects estimated with propensity-screened outcome regression;
  unsupported actions fall back to the configured model.
- Two thousand trajectory bootstrap replicates provide the one-sided 95% upper bound.
- Cost includes estimated uncached input, cached prefix reads, and every observable
  output. A switch resets the accumulated cache. Tokens are serialized JSON chars/4,
  never measured usage.
- Primary proxy evaluation excludes terminal assistant language because each
  trajectory's final output is unobservable under the challenge premise. A separately
  named sensitivity adds that language back.
- Behavior-adjusted cost uses training-only, partially pooled model × opening-intent ×
  initial-size estimates for observable output length. Fixed-shape cost is reported
  alongside it.

The primary automatic failure proxy combines an unresolved observable tool error and
repeated identical-action loops. It is not measured task quality. The terminal-language
sensitivity is also a proxy, not a label.
The organizer-held-out split is not present yet, so public-validation numbers must not
be presented as final held-out results.

## Reproduce

```bash
python scripts/routemaxxing.py export/
python -m pytest -q
python -m explorer
```

Generated evidence is under `results/routemaxxing/`. `ablation_comparison.csv` contains
the logged, heuristic, current, ablation, final, and conditional-adaptive rows;
`training_sweep.csv` records every training-only knob; `proposal_rankings.csv` exposes
the dollar/risk ordering; and `propensity_support.json` documents Luna's exclusion. The
model- and context-specific observable-output estimates are in
`output_length_sensitivity.json`. The explorer-compatible route is
`results/routemaxxing.jsonl`; the six-slide deck is
`presentation-routemaxxing.html`.
