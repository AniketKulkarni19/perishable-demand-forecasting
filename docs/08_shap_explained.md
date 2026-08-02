# Phase 8 — SHAP: Understanding *Why* the Model Predicts What It Predicts

**Notebook:** `notebooks/08_shap.ipynb`

> Written with extra explanation, since SHAP is conceptually unfamiliar territory. Read this section slowly on a re-read; the concept is worth cementing properly.

---

## Part 1 — What problem does SHAP solve?

A gradient boosting model is 400 decision trees. Asking "why did it predict 4.2 units for store 11, item 108831, on July 20th?" has no simple answer — the prediction is the sum of 400 separate tree outputs, each splitting on different features at different thresholds.

You can't read it. You can't explain it to a demand planner. And if a planner won't trust a number they can't interrogate, they won't use it — which means the model delivers zero value regardless of its WAPE.

**SHAP answers exactly this question, per prediction.**

---

## Part 2 — The core concept, by analogy

Imagine a sales team of five people closes $500,000 in a quarter. The company's baseline expectation — what any random team would close — is $300,000.

The team beat expectations by $200,000. **How much did each person contribute?**

You can't just look at individual sales totals, because people work together — a deal one person sourced might have been closed by another. The fair approach is to ask: *how much did the outcome change when this person was involved versus not?*

That's exactly what SHAP does, but with features instead of salespeople.

- **Base value** = the average prediction across all data (the $300,000 baseline)
- **SHAP value for a feature** = how much *that feature* pushed *this particular prediction* away from the base value
- All SHAP values sum to the difference between the base value and the actual prediction

The underlying math is **Shapley values**, from cooperative game theory (Lloyd Shapley, 1953 — he won a Nobel for related work). It's the mathematically unique way to fairly distribute credit among contributors, satisfying properties like "contributors who add nothing get zero" and "identical contributors get identical credit."

---

## Part 3 — The additivity property (and why we verified it)

SHAP's defining guarantee:

```
prediction = base_value + sum(all SHAP values for that row)
```

Exactly. No residual, no approximation error.

### Our verification

```python
check = base_value + shap_values[0].sum()
actual_pred = model.predict(X.iloc[[0]])[0]
```

```
Base value (log space): 1.4380  →  3.21 units
Reconstruction check: 3.368286 vs 3.368286   ✓
```

**Six-decimal match.** This isn't ceremony — it's proof that the values are genuine TreeSHAP and not something approximate. If those numbers had diverged, every interpretation downstream would be unreliable.

### Reading a single prediction

For the first sampled row:
```
Base value:              1.4380  (what the model predicts knowing nothing)
+ contribution of roll_mean_14
+ contribution of onpromotion
+ contribution of dayofweek
+ ... (all 43 features)
= 3.3683  (the actual prediction, in log space)
```

Every prediction decomposes exactly this way. That's what makes SHAP explainable rather than just suggestive.

---

## Part 4 — Two importance measures, two different questions

We now have importance rankings from two sources, and **they disagree**:

| Feature | Gain % (Phase 6) | SHAP % (Phase 8) |
|---|---|---|
| roll_mean_14 | 50.93 | 20.95 |
| roll_mean_28 | 22.11 | 17.48 |
| **onpromotion** | **1.59** ⬅ 9th | **10.66** ⬅ 3rd |
| roll_mean_7 | 7.26 | 7.25 |
| **dayofweek** | **1.54** | **6.37** |
| lag_1 | 1.68 | 5.89 |

### Why they differ

**Gain** measures *total loss reduction during training*, summed across every split using that feature. It answers: **"how much did this feature help the model learn?"**

**SHAP** measures *impact on individual predictions*, averaged across rows. It answers: **"how much does this feature move predictions in practice?"**

### The `onpromotion` case — a concrete illustration

Promotions occur on a **small fraction of rows**. So:

- **By gain:** total loss reduction across all training is modest, because it only applies to a few rows → ranks 9th
- **By SHAP:** *when* a promotion occurs, it moves the prediction enormously (+0.3 to +1.0 in log space) → ranks 3rd

**Analogy:** a fire alarm rarely goes off. Measured by "how often does it change your behaviour," it looks unimportant. Measured by "when it does go off, how much does it change your behaviour," it's critical.

**The practical lesson:** relying only on gain would have led to underrating the model's third-most-influential feature. Always look at both.

---

## Part 5 — Reading the beeswarm plot

Each dot is one prediction. Position on the x-axis = that feature's SHAP contribution. Colour = the feature's *value* (red = high, blue = low).

### What our plot shows

**`onpromotion` — the cleanest signal in the model.**
Perfect separation: a tight blue cluster at zero (no promotion → no effect) and a distinct red band pushing +0.3 to +1.0. Nothing else in the plot is that clean.

*Interpretation:* promotions reliably and substantially increase demand. This visually validates the Phase 2 decision to discard two years of data in order to keep promotion data intact — that decision now has proof, not just reasoning.

**`is_weekend` — binary and one-directional.**
Solid red block at ~+0.15, blue at zero. Weekends increase demand, consistently.

**`dayofweek` — the weekly cycle made visible.**
Blue (low values = Monday/Tuesday) pushes predictions *down* to −0.5. Red (high values = Saturday/Sunday) pushes *up*.

This explains a number from the importance table: `dayofweek` had **mean_shap ≈ +0.000028** (essentially zero) but **mean_abs_shap = 0.0766** (high). It pushes up as often as it pushes down, so the net cancels while the magnitude is large. That is precisely what a cyclical feature should look like.

**Rolling means — mostly blue with long right tails.**
Most rows have low recent averages (small demand values dominate the dataset), pushing predictions down. Occasional high-value rows push hard upward. This is the right-skewed demand distribution showing through — and it's the same skew that justified the `log1p` target transform in Phase 6.

### An honest caveat about this plot

The rolling means appear almost uniformly blue because the colour scale uses **min–max normalization**, which gets crushed by extreme outliers. A handful of very large values compress everything else into the low end of the scale.

The *pattern* is real, but the colouring understates the variation. A percentile-based colour scale would show it better. Worth noting rather than presenting the visualization as flawless.

---

## Part 6 — The dependency problem (and the workaround)

### What happened

The `shap` package would not install:

```
numba==0.61.2 depends on numpy>=1.24,<2.3
your project depends on numpy>=2.5.1
→ requirements are unsatisfiable
```

SHAP requires `numba`, which caps at numpy < 2.3. This project runs numpy 2.5.1 because pandas 3.0 requires it. **Irreconcilable** — and downgrading numpy would break pandas and force re-running the entire pipeline.

Earlier attempts also hit release-candidate versions with no prebuilt wheels, which tried to compile LLVM bindings from source and demanded CMake.

### The solution

**LightGBM implements TreeSHAP natively:**

```python
shap_raw = model.predict(X, pred_contrib=True)
shap_values = shap_raw[:, :-1]   # per-feature contributions
base_value  = shap_raw[0, -1]    # last column is the base value
```

Identical values, zero extra dependencies. What's lost is only SHAP's plotting convenience functions — which are matplotlib wrappers, reproducible in ~15 lines.

### Why this is a good decision, not a compromise

Installing CMake and compiling LLVM bindings on a 2019 Intel Mac would have cost hours for **zero analytical gain**. The values are the same either way.

*"I computed SHAP values using LightGBM's built-in TreeSHAP implementation"* is a completely normal sentence — and arguably demonstrates understanding of what SHAP is, rather than just knowing which library to call.

**Recognising when a dependency fight isn't worth having is a real engineering skill.**

### A small bug worth remembering

The first plotting attempt crashed:
```
TypeError: numpy boolean subtract, the `-` operator, is not supported
```

Cause: `pd.api.types.is_numeric_dtype()` returns `True` for booleans, so the min–max normalization tried to subtract boolean values. Fixed by checking `is_bool_dtype()` **before** `is_numeric_dtype()`, since bool is a subtype of numeric in pandas.

Order of type checks matters when the categories overlap.

---

## Part 7 — Why this phase exists at all

Three concrete reasons, worth being able to state:

**1. Trust.** A demand planner will not act on a number they cannot interrogate. SHAP lets you say *"the forecast is high because the item is on promotion and it's a Saturday"* — which is actionable, and which a WAPE score is not.

**2. Debugging.** If a feature that should matter shows near-zero SHAP, something is wrong — a broken join, a leakage guard that zeroed a column, a dead feature. SHAP surfaces these silently-broken features that metrics alone would hide.

**3. Feature pruning.** Features contributing nothing can be removed, making the model faster and simpler. In Phase 6, `is_workday_makeup` had exactly 0.00 gain — never used in a single split. That's a feature to delete.

---

## Key takeaways

- *"SHAP decomposes each individual prediction into a base value plus one contribution per feature, and those contributions sum exactly to the prediction. I verified that additivity property to six decimal places before trusting any of the analysis."*
- *"Gain and SHAP importance disagreed meaningfully. `onpromotion` ranked 9th by gain but 3rd by SHAP, because promotions occur on few rows but move predictions enormously when they do. Gain measures total training contribution; SHAP measures per-prediction impact. Relying on one alone would have led me to underrate a key feature."*
- *"The `dayofweek` feature had near-zero mean SHAP but high mean absolute SHAP — it pushes predictions up on weekends and down on weekdays, so the net cancels. That's the signature of a cyclical feature, and it's only visible when you look at both statistics."*
- *"The SHAP package wouldn't install due to an irreconcilable numpy conflict with pandas 3.0. Rather than spend hours compiling LLVM bindings from source, I used LightGBM's native TreeSHAP implementation, which produces identical values. Knowing when a dependency fight isn't worth having is part of the job."*
- *"SHAP matters because a demand planner won't act on a number they can't interrogate. Being able to say 'this forecast is high because the item is on promotion and it's a Saturday' is what makes a model usable rather than just accurate."*
