# Phase 1 Evaluation Results
> Generated: 2026-06-13 14:36:10

## Dataset
- **Source**: Simulated pressure vessel (ODE-based)
- **Fault types**: seal_degradation, heater_drift, blockage
- **Split**: train/val/test by run_id

## Detailed Results (per model x fault type)

| Model | Fault Type | F1 | Precision | Recall | TTD (s) | Threshold |
|-------|-----------|-----|-----------|--------|---------|-----------|
| PINN | seal_degradation | 0.9679 | 0.9956 | 0.9417 | 8.0 | 2.8802 |
| IsolationForest | seal_degradation | 0.8625 | 0.7869 | 0.9542 | 0.0 | -0.0457 |
| LSTM-AE | seal_degradation | 0.9565 | 1.0000 | 0.9167 | 11.0 | 0.9899 |
| PINN | heater_drift | 0.6338 | 0.6520 | 0.6167 | 0.0 | 1.5636 |
| IsolationForest | heater_drift | 0.6536 | 0.6849 | 0.6250 | 0.0 | -0.0528 |
| LSTM-AE | heater_drift | 0.6005 | 0.8421 | 0.4667 | 118.0 | 0.0022 |
| PINN | blockage | 0.9544 | 0.9955 | 0.9167 | 11.0 | 3.2365 |
| IsolationForest | blockage | 0.8699 | 0.8145 | 0.9333 | 0.0 | -0.0386 |
| LSTM-AE | blockage | 0.9357 | 1.0000 | 0.8792 | 15.5 | 2.3461 |

## Summary (averaged across fault types)

| Model | Avg F1 | Avg Precision | Avg Recall | Avg TTD (s) |
|-------|--------|--------------|------------|-------------|
| PINN | 0.8521 | 0.8810 | 0.8250 | 6.3 |
| IsolationForest | 0.7953 | 0.7621 | 0.8375 | 0.0 |
| LSTM-AE | 0.8309 | 0.9474 | 0.7542 | 48.2 |

## Methodology
- **Threshold selection**: Optimal F1-maximizing threshold found on validation set
- **Time-to-detection (TTD)**: Time from true fault onset to first 3 consecutive
  anomaly flags above threshold
- **PINN anomaly score**: Hybrid of prediction deviation + ODE residual on observed data
- **LSTM-AE anomaly score**: Reconstruction error (MSE)
- **IsolationForest anomaly score**: Negated sklearn decision_function

## Notes
- All models trained only on normal operation data
- PINN uses physics-informed loss (data + ODE residual)
- Baselines use purely data-driven approaches
