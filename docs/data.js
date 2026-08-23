window.ROUTEMAXXING_DEMO = {
  "meta": {
    "name": "Routemaxxing Router Lab",
    "split": "public validation",
    "token_estimator": "serialized JSON characters / 4; not measured usage",
    "held_out": "not used for development",
    "data_policy": "aggregate evidence only; no challenge trajectories included"
  },
  "headline": {
    "logged_cost": 88.8784,
    "current_cost": 68.9774,
    "routed_cost": 48.1798,
    "behavior_adjusted_cost": 47.8957,
    "savings_pct": 45.79,
    "failure_upper95_pp": -0.187,
    "coverage_pct": 58.269999999999996,
    "effective_n": 155.4,
    "minimum_propensity": 0.2552
  },
  "pricing": {
    "claude-sonnet-5": [
      2.0,
      0.2,
      10.0
    ],
    "claude-opus-5": [
      5.0,
      0.5,
      25.0
    ],
    "claude-fable-5": [
      10.0,
      1.0,
      50.0
    ],
    "gpt-5.6-luna": [
      0.2,
      0.02,
      1.2
    ],
    "gpt-5.6-terra": [
      2.0,
      0.2,
      12.0
    ],
    "gpt-5.6-sol": [
      5.0,
      0.5,
      30.0
    ]
  },
  "certified_downgrades": {
    "claude-fable-5": "claude-sonnet-5",
    "claude-opus-5": "claude-sonnet-5",
    "gpt-5.6-sol": "gpt-5.6-terra"
  },
  "families": {
    "claude": [
      "claude-sonnet-5",
      "claude-opus-5",
      "claude-fable-5"
    ],
    "gpt": [
      "gpt-5.6-luna",
      "gpt-5.6-terra",
      "gpt-5.6-sol"
    ]
  },
  "support": {
    "minimum_train_n": 20,
    "gpt-5.6-luna": {
      "train_n": 12,
      "eligible": false
    }
  },
  "output_length_estimates": {
    "claude-sonnet-5": 174.24777816253513,
    "gpt-5.6-sol": 290.7783663212085,
    "gpt-5.6-terra": 287.24292404515114,
    "claude-opus-4-8": 197.38442113494054,
    "claude-opus-5": 186.87842558971602,
    "claude-fable-5": 229.94350648498036,
    "gpt-5.6-luna": 310.3690911809129,
    "claude-sonnet-4-6": 191.94461481780473,
    "claude-opus-4-6": 207.64012763831755
  },
  "frontier": [
    {
      "adopt_frac": 0.0,
      "cost_usd_est": 88.8784,
      "behavior_adjusted_cost_usd_est": 88.8784,
      "savings_pct": 0.0,
      "failure_delta": 0.0,
      "failure_upper95": 0.0,
      "coverage": 0.0,
      "effective_n": 278,
      "selected": "False"
    },
    {
      "adopt_frac": 0.05,
      "cost_usd_est": 86.3821,
      "behavior_adjusted_cost_usd_est": 86.3344,
      "savings_pct": 2.81,
      "failure_delta": -0.00033,
      "failure_upper95": -0.00012,
      "coverage": 0.0216,
      "effective_n": 6.0,
      "selected": "False"
    },
    {
      "adopt_frac": 0.1,
      "cost_usd_est": 83.43,
      "behavior_adjusted_cost_usd_est": 83.3557,
      "savings_pct": 6.13,
      "failure_delta": -0.00038,
      "failure_upper95": -0.00016,
      "coverage": 0.0683,
      "effective_n": 19.0,
      "selected": "False"
    },
    {
      "adopt_frac": 0.15,
      "cost_usd_est": 81.1584,
      "behavior_adjusted_cost_usd_est": 81.0463,
      "savings_pct": 8.69,
      "failure_delta": -0.00065,
      "failure_upper95": -0.00034,
      "coverage": 0.0863,
      "effective_n": 23.8,
      "selected": "False"
    },
    {
      "adopt_frac": 0.2,
      "cost_usd_est": 79.1638,
      "behavior_adjusted_cost_usd_est": 79.0312,
      "savings_pct": 10.93,
      "failure_delta": -0.00106,
      "failure_upper95": -0.00065,
      "coverage": 0.1223,
      "effective_n": 33.4,
      "selected": "False"
    },
    {
      "adopt_frac": 0.25,
      "cost_usd_est": 78.8878,
      "behavior_adjusted_cost_usd_est": 78.756,
      "savings_pct": 11.24,
      "failure_delta": -0.00142,
      "failure_upper95": -0.00094,
      "coverage": 0.1403,
      "effective_n": 38.4,
      "selected": "False"
    },
    {
      "adopt_frac": 0.3,
      "cost_usd_est": 75.3322,
      "behavior_adjusted_cost_usd_est": 75.1454,
      "savings_pct": 15.24,
      "failure_delta": -0.00395,
      "failure_upper95": -0.00313,
      "coverage": 0.2374,
      "effective_n": 64.6,
      "selected": "False"
    },
    {
      "adopt_frac": 0.35,
      "cost_usd_est": 75.3322,
      "behavior_adjusted_cost_usd_est": 75.1454,
      "savings_pct": 15.24,
      "failure_delta": -0.00395,
      "failure_upper95": -0.00313,
      "coverage": 0.2374,
      "effective_n": 64.6,
      "selected": "False"
    },
    {
      "adopt_frac": 0.4,
      "cost_usd_est": 75.3322,
      "behavior_adjusted_cost_usd_est": 75.1454,
      "savings_pct": 15.24,
      "failure_delta": -0.00395,
      "failure_upper95": -0.0031,
      "coverage": 0.2374,
      "effective_n": 64.6,
      "selected": "False"
    },
    {
      "adopt_frac": 0.45,
      "cost_usd_est": 74.657,
      "behavior_adjusted_cost_usd_est": 74.4564,
      "savings_pct": 16.0,
      "failure_delta": -0.00392,
      "failure_upper95": -0.00307,
      "coverage": 0.241,
      "effective_n": 65.6,
      "selected": "False"
    },
    {
      "adopt_frac": 0.5,
      "cost_usd_est": 67.007,
      "behavior_adjusted_cost_usd_est": 66.7358,
      "savings_pct": 24.61,
      "failure_delta": -0.00575,
      "failure_upper95": -0.00486,
      "coverage": 0.3849,
      "effective_n": 104.5,
      "selected": "False"
    },
    {
      "adopt_frac": 0.55,
      "cost_usd_est": 67.007,
      "behavior_adjusted_cost_usd_est": 66.7358,
      "savings_pct": 24.61,
      "failure_delta": -0.00575,
      "failure_upper95": -0.00484,
      "coverage": 0.3849,
      "effective_n": 104.5,
      "selected": "False"
    },
    {
      "adopt_frac": 0.6,
      "cost_usd_est": 67.007,
      "behavior_adjusted_cost_usd_est": 66.7358,
      "savings_pct": 24.61,
      "failure_delta": -0.00575,
      "failure_upper95": -0.0048,
      "coverage": 0.3849,
      "effective_n": 104.5,
      "selected": "False"
    },
    {
      "adopt_frac": 0.65,
      "cost_usd_est": 67.007,
      "behavior_adjusted_cost_usd_est": 66.7358,
      "savings_pct": 24.61,
      "failure_delta": -0.00575,
      "failure_upper95": -0.00483,
      "coverage": 0.3849,
      "effective_n": 104.5,
      "selected": "False"
    },
    {
      "adopt_frac": 0.7,
      "cost_usd_est": 67.007,
      "behavior_adjusted_cost_usd_est": 66.7358,
      "savings_pct": 24.61,
      "failure_delta": -0.00575,
      "failure_upper95": -0.00481,
      "coverage": 0.3849,
      "effective_n": 104.5,
      "selected": "False"
    },
    {
      "adopt_frac": 0.75,
      "cost_usd_est": 65.257,
      "behavior_adjusted_cost_usd_est": 64.9581,
      "savings_pct": 26.58,
      "failure_delta": -0.00568,
      "failure_upper95": -0.00476,
      "coverage": 0.3957,
      "effective_n": 107.5,
      "selected": "False"
    },
    {
      "adopt_frac": 0.8,
      "cost_usd_est": 60.11,
      "behavior_adjusted_cost_usd_est": 59.7592,
      "savings_pct": 32.37,
      "failure_delta": -0.00551,
      "failure_upper95": -0.00454,
      "coverage": 0.4173,
      "effective_n": 113.0,
      "selected": "False"
    },
    {
      "adopt_frac": 0.85,
      "cost_usd_est": 58.0202,
      "behavior_adjusted_cost_usd_est": 57.6378,
      "savings_pct": 34.72,
      "failure_delta": -0.0053,
      "failure_upper95": -0.00436,
      "coverage": 0.4424,
      "effective_n": 119.7,
      "selected": "False"
    },
    {
      "adopt_frac": 0.9,
      "cost_usd_est": 55.7939,
      "behavior_adjusted_cost_usd_est": 55.4247,
      "savings_pct": 37.22,
      "failure_delta": -0.00477,
      "failure_upper95": -0.00376,
      "coverage": 0.4784,
      "effective_n": 128.6,
      "selected": "False"
    },
    {
      "adopt_frac": 0.95,
      "cost_usd_est": 53.4774,
      "behavior_adjusted_cost_usd_est": 53.1214,
      "savings_pct": 39.83,
      "failure_delta": -0.00405,
      "failure_upper95": -0.00294,
      "coverage": 0.536,
      "effective_n": 143.2,
      "selected": "False"
    },
    {
      "adopt_frac": 1.0,
      "cost_usd_est": 48.1798,
      "behavior_adjusted_cost_usd_est": 47.8957,
      "savings_pct": 45.79,
      "failure_delta": -0.00305,
      "failure_upper95": -0.00187,
      "coverage": 0.5827,
      "effective_n": 155.4,
      "selected": "True"
    }
  ],
  "comparison": [
    {
      "policy": "logged_behavior",
      "cost_usd_est": 88.8784,
      "behavior_adjusted_cost_usd_est": 88.8784,
      "savings_pct": 0.0,
      "behavior_adjusted_savings_pct": 0.0,
      "failure_upper95": 0.0,
      "coverage": 0.0,
      "effective_n": 278,
      "note": null
    },
    {
      "policy": "corrected_heuristic_baseline",
      "cost_usd_est": 74.6847,
      "behavior_adjusted_cost_usd_est": 74.635,
      "savings_pct": 15.97,
      "behavior_adjusted_savings_pct": 16.03,
      "failure_upper95": -7e-05,
      "coverage": 0.295,
      "effective_n": 77.9,
      "note": "call-zero prompt-size heuristic; family/support guarded"
    },
    {
      "policy": "current_routemaxxing_certified",
      "cost_usd_est": 68.9774,
      "behavior_adjusted_cost_usd_est": 68.8262,
      "savings_pct": 22.39,
      "behavior_adjusted_savings_pct": 22.56,
      "failure_upper95": -0.00192,
      "coverage": 0.4173,
      "effective_n": 112.9,
      "note": null
    },
    {
      "policy": "final_improved_certified",
      "cost_usd_est": 48.1798,
      "behavior_adjusted_cost_usd_est": 47.8957,
      "savings_pct": 45.79,
      "behavior_adjusted_savings_pct": 46.11,
      "failure_upper95": -0.00187,
      "coverage": 0.5827,
      "effective_n": 155.4,
      "note": null
    },
    {
      "policy": "adaptive_conditional",
      "cost_usd_est": 79.5307,
      "behavior_adjusted_cost_usd_est": 79.4398,
      "savings_pct": 10.52,
      "behavior_adjusted_savings_pct": 10.62,
      "failure_upper95": null,
      "coverage": 0.5176,
      "effective_n": null,
      "note": "conditional on logged states; no end-to-end failure estimate"
    }
  ],
  "constraints": [
    "One model for the complete certified trajectory",
    "Never upgrade above the configured default",
    "Preserve the tool-dialect model family",
    "Require positive fixed and behavior-adjusted savings",
    "Unsupported actions fall back to the default",
    "Primary proxy upper95 must remain at or below +2 percentage points"
  ],
  "known_weakness": "Proxy failure is not measured task quality. Outcome regression can fail under residual confounding, and adaptive state transitions are conditional on the logged trajectory rather than identified end to end."
};
