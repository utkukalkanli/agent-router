# Contextual cost router

The implementation is in `scripts/routers/v2_oracle.py`. It is additive: none of the
starter scripts are changed.

## Run

```bash
python3.13 scripts/routers/v2_oracle.py export/
```

This writes versioned routes plus `results/oracle_router/comparison.csv`,
`grouped_validation.csv`, `sensitivity.csv`, `frontier.csv`, `calibration.json`,
and `REPORT.md`.

For online use, fit once on historical rows and call the router on the first request:

```python
from scripts.routers.v2_oracle import fit_router

router = fit_router(historical_complete_rows, version="router_v4_conservative")
decision = router.route(first_request)
model = decision["model"]
per_model_estimates = decision["estimates"]
```

Each candidate estimate contains conditional-mean future uncached input, cached input,
output tokens, official-price spend, a separate typical-length `total_tokens_p50`, the
execution-quality proxy, uncertainty, and matched support. The online feature extractor
stops before the first model-produced item.

## Important correction to the starter assumption

In the supplied real chunk, each of the 1,000 JSONL rows is a complete trajectory.
Model-produced runs inside the row reconstruct 10,845 billed calls. Grouping rows by
opening text incorrectly produces 953 groups, including 19 mixed-model collisions.

Call decomposition recovers roughly 2.62M estimated output tokens. On the corrected,
cache-aware input+output ledger, logged spend is $278.36; treating each row as one
input-only request materially undercounts it. Every token number is still a chars/4
estimate because provider usage is absent.

## Versions

| Version | Idea | Estimated spend | Saving vs logged | Outcome proxy |
|---|---|---:|---:|---:|
| Corrected starter | Final-size `<15k` switch; future-leaking reference | $265.85 | 4.5% | 0.9891 |
| v2 workload | Predict workload; cheap below median, capable default above | $246.26 | 11.5% | 0.9909 |
| v3 matched | Cheapest matched model within 0.8 proxy points of best | $184.75 | 33.6% | 0.9861 |
| v4 conservative | Quality tolerance + uncertainty + 95% support radius | $138.00 | 50.4% | 0.9890 |
| v5 near-oracle | Aggressive same-family cost optimum | $120.48 | 56.7% | 0.9890 |
| v6 analogue | v4 plus aggressive routing for recurring task templates | $136.77 | 50.9% | 0.9879 |
| Same-trace floor | Cheapest same-family rate on realized quantities | $100.85 | 63.8% | unknowable |

v5 captures about 89% of the gap from logged spend to the same-trace cost floor, but
its GPT decision rests on only 20 logged Luna trajectories. v4 is the recommended
defensible policy and is 48.1% cheaper than the corrected starter. Its paired quality
difference from logged is +0.0004 with a 95% bootstrap interval of
[-0.0041, +0.0048]: no detected change, not proof of equivalence.

## Evaluation

- Five-fold cross-fitting: a trajectory is never used to predict itself.
- A second task-grouped five-fold check prevents recurring templates crossing folds.
- Transparent lexical/structural nearest-neighbour matching.
- Training-only recurring-task analogue calibration.
- Local behavior propensities and a 0.05-clipped doubly robust estimator.
- Propensity-floor sensitivity from 0.02 through 0.20 ($138.00–$139.12 for v4).
- Whole-trajectory, same-harness-family routing; no mid-task cache reset.
- High-precision tool failure/recovery proxy instead of the placeholder “kept model.”
- Paired bootstrap intervals for policy-minus-logged cost and quality.

Factual-action aggregate calibration is 0.908x actual cost and 0.880x actual expected
tokens. Expected cost/length remains heavy-tailed (cost MdAPE 64.4%, expected-token
MdAPE 98.2%). The separate p50 token forecast has 49.6% MdAPE overall and 28.0% for
known recurring templates. These are routing estimates, not precise invoices.

## Known limitations

Model assignment is observational, GPT/Claude harnesses have no useful cross-family
overlap, counterfactual models can change the trajectory itself, and the quality target
is an execution-completion proxy rather than a human label. The next export chunk must
be treated as untouched validation; selecting another frontier point on it would turn
it into training data.
