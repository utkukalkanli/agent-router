(() => {
  "use strict";

  const data = window.ROUTEMAXXING_DEMO;
  if (!data) throw new Error("Demo data did not load");

  const $ = (id) => document.getElementById(id);
  const usd = (value) => `$${Number(value).toFixed(2)}`;
  const pct = (value, digits = 1) => `${Number(value).toFixed(digits)}%`;
  const pp = (value) => `${Number(value) >= 0 ? "+" : "−"}${Math.abs(Number(value)).toFixed(2)} pp`;
  const formatModel = (model) => model || "—";
  const familyOf = (model) => model.startsWith("claude") ? "claude" : "gpt";

  const h = data.headline;
  $("hero-cost").textContent = usd(h.routed_cost);
  $("hero-cost-note").textContent = `${usd(h.logged_cost)} logged · ${usd(h.current_cost)} current router`;
  $("hero-savings").textContent = pct(h.savings_pct, 2);
  $("hero-upper").textContent = pp(h.failure_upper95_pp);
  $("hero-coverage").textContent = pct(h.coverage_pct, 1);
  $("hero-support").textContent = `ESS ${h.effective_n.toFixed(1)} · propensity ≥ ${h.minimum_propensity.toFixed(3)}`;

  const modelSelect = $("default-model");
  Object.keys(data.pricing).forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = model === "claude-opus-5";
    modelSelect.append(option);
  });

  function route(model, dialect) {
    const family = familyOf(model);
    if (family !== dialect) {
      return {
        selected: model,
        kind: "fallback",
        label: "Default fallback",
        reason: "The selected tool dialect and default model family disagree. Certified mode refuses a cross-family counterfactual.",
        codes: ["tool-dialect-mismatch", "unsupported-action-fallback"],
      };
    }
    const selected = data.certified_downgrades[model];
    if (!selected) {
      const luna = model === "gpt-5.6-luna";
      return {
        selected: model,
        kind: "kept",
        label: "Keep default",
        reason: luna
          ? "Luna has only 12 training trajectories versus 20 required. It may remain a configured default, but the certified policy never introduces a new Luna action."
          : "No strictly cheaper action in common support satisfies the certified policy. The configured default is preserved.",
        codes: luna ? ["below-minimum-support", "configured-default-kept"] : ["no-positive-supported-downgrade", "configured-default-kept"],
      };
    }
    return {
      selected,
      kind: "downgrade",
      label: "Certified downgrade",
      reason: "This same-family downgrade has common propensity support and positive predicted fixed-shape and behavior-adjusted savings.",
      codes: ["same-family", "propensity-supported", "positive-savings", "one-model-trajectory"],
    };
  }

  function estimate(model, shape, outputScale = 1) {
    const [uncachedRate, cachedRate, outputRate] = data.pricing[model];
    let cost = 0;
    for (let index = 0; index < shape.calls; index += 1) {
      const billed = shape.opening + shape.schemas + index * shape.growth;
      const cached = index === 0 ? 0 : shape.opening + shape.schemas + (index - 1) * shape.growth;
      const outputTokens = index === shape.calls - 1 ? 0 : shape.output * outputScale;
      cost += ((billed - cached) * uncachedRate + cached * cachedRate + outputTokens * outputRate) / 1e6;
    }
    return cost;
  }

  function updateRouter() {
    const model = modelSelect.value;
    const dialect = $("dialect").value;
    const decision = route(model, dialect);
    const shape = {
      calls: Math.max(1, Number($("calls").value) || 1),
      opening: Math.max(0, Number($("opening").value) || 0),
      schemas: Math.max(0, Number($("schemas").value) || 0),
      growth: Math.max(0, Number($("growth").value) || 0),
      output: Math.max(0, Number($("output").value) || 0),
    };
    const logged = estimate(model, shape);
    const routed = estimate(decision.selected, shape);
    const outputRatio = Math.min(2, Math.max(.5,
      (data.output_length_estimates[decision.selected] || 1) /
      (data.output_length_estimates[model] || 1)
    ));
    const adjusted = estimate(decision.selected, shape, outputRatio);
    const saving = Math.max(0, logged - routed);
    const savingPct = logged ? saving / logged * 100 : 0;
    const auditCodes = [
      `intent-${$("intent").value.toLowerCase().replace(/[^a-z]+/g, "-").replace(/(^-|-$)/g, "")}`,
      `trigger-${$("trigger").value.toLowerCase()}`,
    ];
    if ($("has-image").checked) auditCodes.push("initial-image");

    $("logged-model").textContent = formatModel(model);
    $("selected-model").textContent = formatModel(decision.selected);
    const badge = $("decision-badge");
    badge.textContent = decision.label;
    badge.className = `decision-badge ${decision.kind}`;
    $("decision-reason").textContent = decision.reason;
    $("logged-cost").textContent = usd(logged);
    $("routed-cost").textContent = usd(routed);
    $("saving-dollar").textContent = `${usd(saving)} saved`;
    $("saving-percent").textContent = pct(savingPct, 1);
    $("saving-fill").style.width = `${Math.min(100, savingPct)}%`;
    $("adjusted-cost").textContent = usd(adjusted);
    $("adjusted-note").textContent = decision.selected === model
      ? "No model change; fixed and adjusted shapes are identical."
      : `${outputRatio.toFixed(2)}× selected/default observable-output length ratio, partially pooled on training data.`;
    $("reason-codes").replaceChildren(...decision.codes.concat(auditCodes).map((code) => {
      const span = document.createElement("span");
      span.className = "chip";
      span.textContent = code;
      return span;
    }));
  }

  modelSelect.addEventListener("change", () => {
    $("dialect").value = familyOf(modelSelect.value);
    updateRouter();
  });
  $("router-form").addEventListener("input", updateRouter);
  updateRouter();

  function drawFrontier() {
    const svg = $("frontier-chart");
    const ns = "http://www.w3.org/2000/svg";
    const rows = data.frontier;
    const bounds = { left: 72, right: 24, top: 28, bottom: 58, width: 760, height: 390 };
    const costs = rows.map((row) => row.cost_usd_est);
    const uppers = rows.map((row) => row.failure_upper95 * 100);
    const xMin = Math.min(...costs) - 2;
    const xMax = Math.max(...costs) + 2;
    const yMin = Math.min(...uppers) - .08;
    const yMax = Math.max(.05, ...uppers) + .08;
    const x = (value) => bounds.left + (value - xMin) / (xMax - xMin) * (bounds.width - bounds.left - bounds.right);
    const y = (value) => bounds.height - bounds.bottom - (value - yMin) / (yMax - yMin) * (bounds.height - bounds.top - bounds.bottom);
    const node = (name, attributes = {}, text = "") => {
      const element = document.createElementNS(ns, name);
      Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
      if (text) element.textContent = text;
      return element;
    };
    for (let index = 0; index <= 4; index += 1) {
      const value = yMin + (yMax - yMin) * index / 4;
      svg.append(node("line", { x1: bounds.left, x2: bounds.width - bounds.right, y1: y(value), y2: y(value), stroke: "#ddd7ff" }));
      svg.append(node("text", { x: bounds.left - 10, y: y(value) + 4, "text-anchor": "end", fill: "#68647a", "font-size": 11 }, `${value.toFixed(2)} pp`));
    }
    for (let index = 0; index <= 4; index += 1) {
      const value = xMin + (xMax - xMin) * index / 4;
      svg.append(node("text", { x: x(value), y: bounds.height - 28, "text-anchor": "middle", fill: "#68647a", "font-size": 11 }, usd(value)));
    }
    svg.append(node("path", { d: rows.map((row, index) => `${index ? "L" : "M"}${x(row.cost_usd_est)},${y(row.failure_upper95 * 100)}`).join(" "), fill: "none", stroke: "#6748fd", "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    rows.forEach((row) => {
      const circle = node("circle", { cx: x(row.cost_usd_est), cy: y(row.failure_upper95 * 100), r: row.selected === "True" || row.selected === true ? 9 : 5, fill: row.selected === "True" || row.selected === true ? "#ffbd9e" : "#6748fd", stroke: row.selected === "True" || row.selected === true ? "#150079" : "#fff", "stroke-width": 2 });
      circle.append(node("title", {}, `${Math.round(row.adopt_frac * 100)}% adoption · ${usd(row.cost_usd_est)} · upper95 ${pp(row.failure_upper95 * 100)}`));
      svg.append(circle);
    });
    svg.append(node("text", { x: (bounds.left + bounds.width - bounds.right) / 2, y: bounds.height - 5, "text-anchor": "middle", fill: "#150079", "font-size": 12, "font-weight": 700 }, "Estimated cache-aware cost"));
    svg.append(node("text", { transform: `translate(18 ${(bounds.top + bounds.height - bounds.bottom) / 2}) rotate(-90)`, "text-anchor": "middle", fill: "#150079", "font-size": 12, "font-weight": 700 }, "One-sided failure-proxy upper95"));
  }
  drawFrontier();

  $("selected-savings-big").textContent = pct(h.savings_pct, 2);
  $("selected-fixed").textContent = usd(h.routed_cost);
  $("selected-adjusted").textContent = usd(h.behavior_adjusted_cost);
  $("selected-ess").textContent = h.effective_n.toFixed(1);
  $("selected-propensity").textContent = h.minimum_propensity.toFixed(3);

  const labels = {
    logged_behavior: "Logged behavior",
    corrected_heuristic_baseline: "Corrected heuristic",
    current_routemaxxing_certified: "Current certified",
    final_improved_certified: "Improved certified",
    adaptive_conditional: "Adaptive (conditional)",
  };
  $("comparison-body").replaceChildren(...data.comparison.map((row) => {
    const tr = document.createElement("tr");
    if (row.policy === "final_improved_certified") tr.className = "final-row";
    const values = [
      labels[row.policy] || row.policy,
      row.cost_usd_est == null ? "—" : usd(row.cost_usd_est),
      row.behavior_adjusted_cost_usd_est == null ? "—" : usd(row.behavior_adjusted_cost_usd_est),
      row.savings_pct == null ? "—" : pct(row.savings_pct, 2),
      row.failure_upper95 == null ? "conditional" : pp(row.failure_upper95 * 100),
      row.coverage == null ? "—" : pct(row.coverage * 100, 1),
      row.effective_n == null ? "—" : Number(row.effective_n).toFixed(1),
    ];
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    });
    return tr;
  }));

  $("constraints").replaceChildren(...data.constraints.map((constraint) => {
    const div = document.createElement("div");
    div.className = "constraint";
    div.textContent = constraint;
    return div;
  }));
  $("weakness-copy").textContent = data.known_weakness;
})();
