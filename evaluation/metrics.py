"""
Evaluation Metrics
==================
Computes F1, precision, recall, and time-to-detection (TTD) for
anomaly detection models on the simulated vessel dataset.

Used by the evaluation pipeline to produce comparison tables.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class EvalResult:
    """Evaluation result for a single model on a single fault type."""
    model_name: str
    fault_type: str
    f1: float
    precision: float
    recall: float
    ttd: float          # time-to-detection in seconds (np.inf if not detected)
    threshold: float     # anomaly threshold used
    n_samples: int


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """
    Find the threshold that maximizes F1 score.

    Parameters
    ----------
    labels : (N,) binary labels (0 = normal, 1 = anomaly)
    scores : (N,) anomaly scores (higher = more anomalous)
    n_thresholds : number of thresholds to try

    Returns
    -------
    best_threshold : float
    best_f1 : float
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1 = 0.0
    best_threshold = thresholds[0]

    for th in thresholds:
        preds = (scores >= th).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    return best_threshold, best_f1


def compute_ttd(
    timestamps: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_consecutive: int = 3,
) -> float:
    """
    Compute time-to-detection (TTD).

    TTD = time from true fault onset to the first sustained anomaly flag
    (score > threshold for n_consecutive consecutive windows).

    Parameters
    ----------
    timestamps : (N,) time values
    labels : (N,) binary labels
    scores : (N,) anomaly scores
    threshold : anomaly threshold
    n_consecutive : number of consecutive windows above threshold required

    Returns
    -------
    ttd : float (seconds), np.inf if fault is never detected
    """
    # Find true fault onset
    fault_indices = np.where(labels == 1)[0]
    if len(fault_indices) == 0:
        return np.inf  # no fault in this segment

    fault_onset_time = timestamps[fault_indices[0]]

    # Find first sustained detection
    above = scores >= threshold
    consecutive_count = 0

    for i in range(len(above)):
        if above[i]:
            consecutive_count += 1
            if consecutive_count >= n_consecutive:
                detection_time = timestamps[i]
                ttd = detection_time - fault_onset_time
                return max(ttd, 0.0)  # clamp at 0 (early detection possible)
        else:
            consecutive_count = 0

    return np.inf  # never detected


def evaluate_model(
    model_name: str,
    fault_type: str,
    timestamps: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    n_consecutive: int = 3,
) -> EvalResult:
    """
    Full evaluation of one model on one fault type.

    Parameters
    ----------
    model_name : identifier for the model
    fault_type : fault scenario name
    timestamps : (N,) time values
    labels : (N,) binary labels
    scores : (N,) anomaly scores
    threshold : if None, finds optimal via F1 maximization
    n_consecutive : consecutive windows for TTD calculation

    Returns
    -------
    EvalResult
    """
    if threshold is None:
        threshold, _ = find_optimal_threshold(labels, scores)

    preds = (scores >= threshold).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    ttd = compute_ttd(timestamps, labels, scores, threshold, n_consecutive)

    return EvalResult(
        model_name=model_name,
        fault_type=fault_type,
        f1=f1,
        precision=prec,
        recall=rec,
        ttd=ttd,
        threshold=threshold,
        n_samples=len(labels),
    )


def format_results_table(results: list[EvalResult]) -> str:
    """
    Format evaluation results as a Markdown table.

    Returns
    -------
    markdown : str
    """
    lines = []
    lines.append("| Model | Fault Type | F1 | Precision | Recall | TTD (s) | Threshold |")
    lines.append("|-------|-----------|-----|-----------|--------|---------|-----------|")

    for r in results:
        ttd_str = f"{r.ttd:.1f}" if r.ttd != np.inf else "INF (not detected)"
        lines.append(
            f"| {r.model_name} | {r.fault_type} | {r.f1:.4f} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {ttd_str} | {r.threshold:.4f} |"
        )

    return "\n".join(lines)


def format_summary_table(results: list[EvalResult]) -> str:
    """
    Format a summary table grouped by model (averaged across fault types).
    """
    from collections import defaultdict

    model_results: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        model_results[r.model_name].append(r)

    lines = []
    lines.append("| Model | Avg F1 | Avg Precision | Avg Recall | Avg TTD (s) |")
    lines.append("|-------|--------|--------------|------------|-------------|")

    for model_name, res_list in model_results.items():
        avg_f1 = np.mean([r.f1 for r in res_list])
        avg_prec = np.mean([r.precision for r in res_list])
        avg_rec = np.mean([r.recall for r in res_list])
        ttds = [r.ttd for r in res_list if r.ttd != np.inf]
        avg_ttd = np.mean(ttds) if ttds else np.inf
        ttd_str = f"{avg_ttd:.1f}" if avg_ttd != np.inf else "INF"

        lines.append(
            f"| {model_name} | {avg_f1:.4f} | {avg_prec:.4f} | "
            f"{avg_rec:.4f} | {ttd_str} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test with synthetic data
    print("=== Metrics Smoke Test ===")

    rng = np.random.default_rng(42)
    n = 200
    timestamps = np.linspace(0, 100, n)
    labels = np.zeros(n, dtype=int)
    labels[100:] = 1  # fault at midpoint

    # Simulated scores: normal ~ 0.1, fault ~ 0.8
    scores = np.where(labels == 0, rng.normal(0.1, 0.05, n), rng.normal(0.8, 0.1, n))

    result = evaluate_model("TestModel", "test_fault", timestamps, labels, scores)
    print(f"F1: {result.f1:.4f}, Precision: {result.precision:.4f}, "
          f"Recall: {result.recall:.4f}, TTD: {result.ttd:.1f}s")

    print("\n" + format_results_table([result]))
    print("\n[OK] Metrics smoke test OK")
