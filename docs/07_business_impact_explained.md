# Phase 7 — Business Impact: Translating WAPE into Dollars

**Notebook:** `notebooks/07_business_impact.ipynb`

> This phase converts a statistical metric into a business one. It is the section that turns "a model with 41% WAPE" into "a system that cuts forecast-related costs by 38%."

---

## 1. The core idea: forecast errors are asymmetric

For perishable goods, over-forecasting and under-forecasting **do not cost the same**.

| Error type | What happens | What it costs |
|---|---|---|
| **Over-forecast** | Order too much → unsold protein spoils | **Full cost of goods** — total loss |
| **Under-forecast** | Order too little → stockout | **Lost margin** — forgone profit only |

Spoiled meat is a complete write-off. A stockout costs you the profit you didn't make, but not the principal you never spent.

**Every standard metric ignores this.** MAE, RMSE, WAPE all treat a 3-unit over-forecast and a 3-unit under-forecast as identical errors. For a business selling fresh protein, they are not remotely identical.

---

## 2. The cost model

```python
UNIT_PRICE = 8.00
MARGIN_PCT = 0.25

COGS   = UNIT_PRICE * (1 - MARGIN_PCT)   # $6.00 — lost when product spoils
MARGIN = UNIT_PRICE * MARGIN_PCT         # $2.00 — lost when you stock out

COST_OVER  = COGS      # $6.00 per excess unit
COST_UNDER = MARGIN    # $2.00 per unit short
```

**Asymmetry ratio: 3.0×.** Over-forecasting is three times more expensive than under-forecasting.

### The cost function

```python
def forecast_cost(actual, forecast, cost_over, cost_under):
    error = forecast - actual
    over  = np.clip(error, 0, None)     # positive errors only
    under = np.clip(-error, 0, None)    # negative errors, sign-flipped
    return over * cost_over + under * cost_under
```

**How the clipping works.** Sign convention: positive error = over-forecast.

`np.clip(error, 0, None)` keeps positives, zeroes negatives → isolates over-forecast amounts.
Flipping the sign first, then clipping → isolates under-forecast amounts.

Every row contributes to exactly one term; the other is zero.

**Worked example:** actual = 10, forecast = 13
- error = +3 → `over` = 3, `under` = 0
- cost = 3 × $6.00 + 0 × $2.00 = **$18.00**

Same magnitude error in the other direction: actual = 13, forecast = 10
- error = −3 → `over` = 0, `under` = 3
- cost = 0 × $6.00 + 3 × $2.00 = **$6.00**

Identical to MAE. Three times different in dollars.

---

## 3. Results on the validation period

```
16-day period, 54 stores, 263 items

Baseline (moving avg): $2,022,885
LightGBM:              $1,370,613
Saving:                  $652,272   (32.2%)
```

### The key observation: 32.2% cost reduction beats the 17.9% WAPE improvement

This is not a coincidence. The decomposition explains it:

| | Over-forecast units | Under-forecast units | Spoilage share of cost |
|---|---|---|---|
| Baseline | 248,356 | 266,374 | 73.7% |
| LightGBM | **131,918** | 289,551 | 57.7% |

**LightGBM nearly halved over-forecast units (248K → 132K) while under-forecasting slightly more (266K → 290K).**

Because over-errors cost 3× more, trading over-errors for under-errors is disproportionately valuable in dollars. The model happened to improve on the **expensive side** of the error distribution.

**WAPE cannot see this.** It weights both directions equally. The cost function reveals value the accuracy metric misses entirely.

---

## 4. Sensitivity analysis — is the result robust?

The dollar figures depend on assumed price and margin. Those are estimates, not Favorita's real numbers. So test whether the conclusion survives different assumptions.

```
  Price   Margin   Ratio   Reduction
$  5.00     15%    5.7x       38.0%
$  5.00     25%    3.0x       32.2%
$  5.00     40%    1.5x       23.7%
$  8.00     15%    5.7x       38.0%
$  8.00     25%    3.0x       32.2%
$  8.00     40%    1.5x       23.7%
$ 12.00     15%    5.7x       38.0%
$ 12.00     25%    3.0x       32.2%
$ 12.00     40%    1.5x       23.7%
```

### Two findings

**Price is completely irrelevant.** Every row at a given margin gives an identical result. This is mathematically necessary — price scales both cost terms equally, so it cancels in the ratio. Price can be dropped from the assumption list entirely, which is one less thing to defend.

**Only the asymmetry ratio matters**, and the improvement holds from **23.7% to 38.0%** across the full range. Even at the least favourable assumption (40% margin, only 1.5× asymmetry), costs still fall by nearly a quarter.

**This is why sensitivity analysis matters.** A single unsupported dollar figure invites scepticism. A range that holds across nine scenarios is a defensible claim.

---

## 5. Bias optimization — the most important finding

Spoilage was still 57.7% of remaining cost, meaning the model systematically over-orders relative to what is *economically* optimal. Since over-errors cost 3×, deliberately shading forecasts downward should reduce total cost.

```python
for f in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]:
    adj = pred * f
```

```
 Factor         Cost    vs 1.0    WAPE
   0.70 $  1,249,361     -8.8%   50.7
   0.75 $  1,237,564     -9.7%   48.3
   0.80 $  1,236,945     -9.8%   46.1   ← optimal
   0.85 $  1,248,667     -8.9%   44.3
   0.90 $  1,274,290     -7.0%   42.8
   0.95 $  1,314,840     -4.1%   41.7
   1.00 $  1,370,613     -0.0%   40.9   ← most accurate
   1.05 $  1,442,055     +5.2%   40.5   ← MOST accurate, MOST expensive
```

### The finding: the cost-optimal forecast is deliberately less accurate

At factor 0.80, cost falls a further 9.8% ($133,668) while **WAPE gets worse** — 40.9 → 46.1.

Look at factor 1.05: it has the **best WAPE in the table (40.5)** and is the **most expensive option**. The two objectives point in opposite directions.

**Why this matters:** it demonstrates that the metric you optimize and the objective you care about are different things. For asymmetric-cost problems, optimizing accuracy actively costs money. Most portfolio projects never make this distinction.

**Caveat to disclose:** the 0.80 factor was selected on validation data. This is a business parameter rather than a model parameter, so it is legitimate — but it must be stated as *"selected on validation, applied unchanged to test."* (It transferred cleanly: 38.8% validation → 38.3% test.)

---

## 6. Results by family

```
                units_sold  cost_baseline  cost_optimal  reduction_pct
DELI              270,728       564,151       362,302          35.8
MEATS             325,969       620,178       348,197          43.9
POULTRY           339,124       685,356       422,555          38.3
PREPARED FOODS     77,276       119,353        82,236          31.1
SEAFOOD            17,300        33,847        21,656          36.0
```

Gains are **consistent across all five families (31%–44%)**, with no single category carrying the result. That is a stronger finding than concentration would be — the improvement is structural rather than an artifact of one well-behaved product group.

MEATS leads at 43.9%.

---

## 7. A caveat worth stating honestly

**The annualized figure ($14.9M) is shaky.** It extrapolates a 16-day July window across a full year. July has no major holidays and no seasonal peaks, so it is unlikely to be representative.

**Lead with the percentage, not the absolute dollars.** The 38% reduction is robust across assumptions; the dollar total depends on both the cost assumptions and the extrapolation.

---

## Key takeaways

- *"Standard metrics treat over- and under-forecasting identically, but for perishable protein they aren't — excess spoils and you lose the full cost of goods, while a stockout only costs you the margin. I built an asymmetric cost function with a 3:1 ratio to measure what the business actually pays."*
- *"The cost reduction of 32% exceeded the WAPE improvement of 18%, because the model happened to cut over-forecast units nearly in half while slightly increasing under-forecasts. Since over-errors cost three times more, that trade is disproportionately valuable — and the accuracy metric can't see it."*
- *"I sensitivity-tested across nine price and margin scenarios. Price cancels out entirely; only the asymmetry ratio matters, and the improvement holds between 24% and 38%. That's a defensible range rather than a single unsupported number."*
- *"The most interesting result is that the cost-optimal forecast is deliberately less accurate. Scaling predictions to 80% worsened WAPE from 40.9 to 46.1 but cut costs another 10%. The most accurate setting in my sweep was also the most expensive one."*
