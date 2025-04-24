# energy_predictor
Train a baseline linear-regression (then perhaps GBM) model that predicts hourly Power Usage Effectiveness or kWh from workload, outside temperature and cooling set-point. Plot “what-if” curves.

## Reading the Plots (`--plot` flag)

| Plot | Observation | Conclusion |
|------|--------------|------------------|
| **1&nbsp;· Daily Energy Consumption – Actual vs Predicted** | Smoothed (24-hour rolling mean) curves for the full 90-day timeline.<br>Blue ≈ Actual · Orange ≈ Model | Curves sit almost on top of each other ⇒ the model captures **long-term load trends**. Any widening gap here would signal *systematic bias* over days/weeks. |
| **2&nbsp;· Hourly Energy Consumption – Test Subset** | Only the 20 % hold-out rows, re-sorted chronologically, plotted hour-by-hour. | Orange track hugs the blue one — consistent with **R² ≈ 0.99** and **MAE ≈ 5 kWh**. Confirms the feature set explains nearly all short-term variance. |
| **3&nbsp;· Δ Energy for Set-Point ± 2 °C**<br>(Outline Histogram) | Two razor-thin spikes at −0.15 kWh and +0.15 kWh. | Because the synthetic PUE formula is linear, every hour shifts by the same amount:<br>**Lowering set-point by 2 °C saves 0.15 kWh h⁻¹ (~0.05 %)**<br>**Raising it adds 0.15 kWh h⁻¹** |
| **4&nbsp;· Prediction Residuals – Test Set** | Error curve (Actual − Predicted) oscillates closely around the 0-line. | Tight band, no visible pattern ⇒ **no systematic bias**; spread ≈ MAE. |

### Overall takeaway
These four visuals demonstrate that the **baseline linear model is already accurate enough for coarse energy-savings experiments**, and that even a modest 2 °C cooling-set-point tweak produces a quantifiable — if small — impact on hourly energy use.
