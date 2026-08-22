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

The certified policy reads the opening task, trigger, toolset, initial size, images, and
nameable intent flags. It chooses the cheapest propensity-supported model whose fitted
failure risk is within two points of the best supported model, but never upgrades above
the configured default. It then keeps that model for the whole trajectory to preserve
the provider cache.

The adaptive policy starts from the certified choice and may escalate once after an
observed tool error, repeated action, or high-risk context growth. It never downgrades
or exceeds the original default. Its result is explicitly conditional on logged states:
a different early model could have generated a different later context.

## Evaluation

- Deterministic 70/30 split grouped by recoverable thread identity, falling back to the
  stable trajectory hash.
- L2 logistic outcome models and multinomial propensity models trained on the public
  training split.
- Downgrade-from-default effects estimated with propensity-screened outcome regression;
  unsupported actions fall back to the configured model.
- Two thousand trajectory bootstrap replicates provide the one-sided 95% upper bound.
- Cost includes estimated uncached input, cached prefix reads, and every observable
  output. A switch resets the accumulated cache. Tokens are serialized JSON chars/4,
  never measured usage.
- The sensitivity columns remove terminal assistant language from the quality proxy and
  adjust output length using training-split model ratios.

The automatic failure proxy combines terminal inability language, an unresolved final
tool error, and repeated identical-action loops. It is not ground-truth task quality.
The organizer-held-out split is not present yet, so public-validation numbers must not
be presented as final held-out results.

## Reproduce

```bash
python scripts/routemaxxing.py export/
python -m pytest -q
python -m explorer
```

Generated evidence is under `results/routemaxxing/`; the explorer-compatible route is
`results/routemaxxing.jsonl`; and the six-slide deck is
`presentation-routemaxxing.html`. The summary JSON is the source of every headline
number in the deck.
