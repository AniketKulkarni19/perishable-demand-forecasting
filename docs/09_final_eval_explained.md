# Phase 9 — Final Evaluation on the Test Set

**Notebook:** `notebooks/09_final_eval.ipynb`

> This is the one time the test set is touched. Everything reported here is an honest estimate of performance on genuinely unseen data.

---

## 1. The discipline

The test set (2017-07-31 → 2017-08-15, 124,039 rows) has been sealed since Phase 4. **Every decision** was made using train and validation only:

| Decision | Made using |
|---|---|
| Feature engineering | Train |
| Hyperparameters | Validation |
| Holiday ablation (keep/drop) | Validation |
| The 0.80 bias factor | Validation |

### The rule

**Whatever the test set produces, it gets reported.** Going back to tune after seeing test results destroys the number's meaning — the test set silently becomes a second validation set, and the reported performance becomes a score that was optimized toward rather than an estimate of future performance.

### What *is* legitimate versus what isn't

| Action | Legitimate? | Why |
|---|---|---|
| Retrain on train+valid with **already-chosen** hyperparameters | ✅ Yes | More data, same configuration. Test untouched |
| Nested cross-validation (inner loop tunes, outer loop evaluates) | ✅ Yes | Rigorous, though heavy |
| Adjust the bias factor because it scored better on test | ❌ No | Test influenced a decision → contaminated |
| Try more boosting rounds because test WAPE looked high | ❌ No | Same problem |

**The test for whether something is legitimate: did the test set influence any decision?** If yes, the number is contaminated.

---

## 2. Results

```
                   model    MAE    RMSE   WAPE   RMSLE
Moving avg 7d (baseline) 3.9162 11.4179  48.60  0.7023
                LightGBM 3.3600  9.1422  41.70  0.5963
    LightGBM + 0.80 bias 3.7382 10.0224  46.39  0.6177
```

```
16-day test period cost:
  Baseline           $1,973,964
  LightGBM           $1,371,719
  LightGBM + bias    $1,218,649

Reduction (LightGBM):        30.5%
Reduction (LightGBM + bias): 38.3%
```

---

## 3. Generalization check — the most important table here

| Metric | Validation | Test | Drift |
|---|---|---|---|
| WAPE (LightGBM) | 40.90 | 41.70 | +0.80 |
| Cost reduction | 32.2% | 30.5% | −1.7 pp |
| **Cost reduction + bias** | **38.8%** | **38.3%** | **−0.5 pp** |

### What this means

**Degradation is minimal and in the expected direction.** Test performance is almost always slightly worse than validation, because validation influenced the choices. A 0.8-point WAPE drift on genuinely unseen data indicates **no meaningful overfitting**.

**The bias factor transferred cleanly** — 38.8% → 38.3%. This was the least certain result, since 0.80 was selected on validation data. Its holding on test means the asymmetric-cost optimization is a real effect, not an artifact of one particular time window.

**If the drift had been large** (say WAPE 40.9 → 48), the correct response would have been to report it honestly as evidence of overfitting — not to go back and re-tune.

---

## 4. Final headline numbers

| Result | Value |
|---|---|
| Test WAPE | **41.70** vs 48.60 baseline |
| Error reduction | **14.2%** |
| Cost reduction (unbiased) | **30.5%** |
| **Cost reduction (bias-optimized)** | **38.3%** |
| Robustness range (margin scenarios) | 24%–38% |

**The core insight:** the cost-optimal forecast has a *worse* WAPE (46.39) than the accuracy-optimal one (41.70). Optimizing for accuracy costs money in an asymmetric-cost setting.

---

## 5. Known limitations (state these; don't hide them)

**The model is undertrained.** Training did not trigger early stopping at 400 rounds — validation RMSE was still improving. More boosting rounds would likely help.

**No classical time-series baselines.** Prophet and SARIMA were skipped. LightGBM comfortably beat the naive baselines, but "did you try classical methods?" currently has the honest answer: no.

**Cost assumptions are estimates.** $8.00 price and 25% margin are plausible but not Favorita's actual figures. Mitigated by sensitivity analysis showing the conclusion holds from 24% to 38%.

**Annualized figures are unreliable.** The 16-day window is July — no major holidays, no seasonal peaks. Extrapolating to a year is directional at best.

**The bias factor was tuned on validation.** Legitimate (it's a business parameter, not a model parameter) and it transferred cleanly to test — but it should always be described as *"selected on validation, applied unchanged to test."*

---

## Key takeaways

- *"I held out the test set from Phase 4 and touched it exactly once, at the end. Every decision — features, hyperparameters, the holiday ablation, the bias factor — was made on validation."*
- *"Test WAPE was 41.70 against 40.90 on validation. A 0.8-point drift on genuinely unseen data means the model generalizes; there's no meaningful overfitting."*
- *"The bias factor was the result I was least confident would transfer, since it was tuned on validation. It held — 38.8% cost reduction on validation, 38.3% on test — which tells me the asymmetric-cost effect is real rather than an artifact of one time window."*
- *"People sometimes suggest evaluating on test and then optimizing further. That converts the test set into a second validation set and makes the reported number meaningless. Retraining on train-plus-validation with already-chosen hyperparameters is fine; letting test results influence a decision is not."*
- *"The model didn't hit early stopping at 400 rounds, so it's undertrained — more boosting is the clearest remaining improvement. I'd rather state that as a known limitation than present the number as fully optimized."*
