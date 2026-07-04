"""Minimal, dependency-free evaluation utilities for a sift selection.

These helpers let you score the set of metrics that :class:`metricsifter.sifter.Sifter`
keeps against a known ground truth -- useful for tuning parameters on your own data
and for guarding against quality regressions in CI. They intentionally add **no**
heavy dependencies (no scikit-learn, no PyRCA); everything is plain set arithmetic.
"""

from dataclasses import dataclass

__all__ = ["SelectionMetrics", "evaluate_selection"]


@dataclass
class SelectionMetrics:
    """Scores for one sift selection against a ground-truth metric set.

    Attributes:
        precision: ``|selected & ground_truth| / |selected|`` -- fraction of
            selected metrics that are truly relevant.
        recall: ``|selected & ground_truth| / |ground_truth|`` -- fraction of the
            relevant metrics that were selected.
        f1: Harmonic mean of ``precision`` and ``recall``.
        reduction_ratio: Fraction of ``all_metrics`` that were **dropped**
            (``1 - kept / total``); ``None`` when ``all_metrics`` was not provided.

    All ratios are defined as ``0.0`` whenever their denominator is zero (e.g.
    empty ``selected``, empty ``ground_truth``, or ``precision + recall == 0``),
    so the metrics are always finite and never raise on degenerate inputs.
    """

    precision: float
    recall: float
    f1: float
    reduction_ratio: float | None = None

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "reduction_ratio": self.reduction_ratio,
        }


def evaluate_selection(
    selected: set[str],
    ground_truth: set[str],
    all_metrics: set[str] | None = None,
) -> SelectionMetrics:
    """Score a selected metric set against a ground truth.

    Args:
        selected: Metric names kept by the sift (e.g. ``result.selected_metrics``).
        ground_truth: Metric names known to be failure-related.
        all_metrics: Optional full set of input metric names. When given, the
            result includes ``reduction_ratio`` (the fraction of the original
            metrics that were dropped).

    Returns:
        SelectionMetrics: precision / recall / f1 (and ``reduction_ratio`` when
        ``all_metrics`` is provided).

    Note:
        Every ratio with a zero denominator is defined as ``0.0`` rather than
        raising: an empty ``selected`` gives ``precision = 0.0``, an empty
        ``ground_truth`` gives ``recall = 0.0``, ``precision + recall == 0`` gives
        ``f1 = 0.0``, and an empty ``all_metrics`` gives ``reduction_ratio = 0.0``.
    """
    selected = set(selected)
    ground_truth = set(ground_truth)

    true_positives = len(selected & ground_truth)
    precision = true_positives / len(selected) if selected else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    reduction_ratio: float | None = None
    if all_metrics is not None:
        all_metrics = set(all_metrics)
        total = len(all_metrics)
        # Count only kept metrics that belong to the original set, so a selection
        # accidentally containing unknown names cannot push the ratio negative.
        kept = len(selected & all_metrics)
        reduction_ratio = (total - kept) / total if total > 0 else 0.0

    return SelectionMetrics(precision=precision, recall=recall, f1=f1, reduction_ratio=reduction_ratio)
