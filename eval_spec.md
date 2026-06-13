# Evaluation Specification

## Metrics
- **F1 score** (anomaly vs normal classification at chosen threshold)
- **Precision / Recall**
- **Time-to-detection (TTD)**: time delta between true fault onset 
  (known from simulation label) and first sustained anomaly flag 
  (score > threshold for N consecutive windows)

## Models Compared
1. PINN physics-residual score
2. LSTM Autoencoder reconstruction error
3. Isolation Forest anomaly score

## Thresholding
- Use validation set to pick threshold maximizing F1 per model
- Report TTD using that threshold on held-out fault scenarios

## Datasets
- **Phase 1**: simulated vessel data (3 fault types, held-out test split)
- **Phase 3**: Tennessee Eastman Process dataset (subset of fault classes)

## Output
- `evaluation/results_phase1.md`: table with F1/Prec/Recall/TTD per model 
  per fault type
- `evaluation/results_phase3.md`: same structure for TEP