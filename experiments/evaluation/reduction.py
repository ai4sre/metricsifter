
def scores_of_synthetic(
    pred_anomalous_metrics: set[str],
    true_root_fault_metrics: set[str],
    true_fault_propagated_metrics: set[str],
    total_metrics: set[str],
) -> dict[str, float]:
    true_anomalous_metrics = true_root_fault_metrics | true_fault_propagated_metrics
    true_normal_metrics = total_metrics - true_anomalous_metrics
    pred_normal_metrics = total_metrics - pred_anomalous_metrics

    root_fault_recall = len(pred_anomalous_metrics & true_root_fault_metrics) / len(true_root_fault_metrics)
    recall = (len(pred_anomalous_metrics & true_anomalous_metrics) / len(true_anomalous_metrics)) if len(true_anomalous_metrics) > 0 else 0.0
    specificity = len(pred_normal_metrics & true_normal_metrics) / len(true_normal_metrics) if len(true_normal_metrics) > 0 else 0.0
    bacc = (recall + specificity) / 2

    return {
        "num_remained": len(pred_anomalous_metrics),
        "num_removed": len(pred_normal_metrics),
        "num_total": len(total_metrics),
        "root_fault_recall": root_fault_recall,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": bacc,
    }


def scores_of_empirical(
    pred_anomalous_metrics: set[str],
    true_root_fault_metrics: set[str],
    total_metrics: set[str],
) -> dict[str, float]:
    true_normal_metrics = total_metrics - true_root_fault_metrics
    pred_normal_metrics = total_metrics - pred_anomalous_metrics

    root_fault_recall = len(pred_anomalous_metrics & true_root_fault_metrics) / len(true_root_fault_metrics)
    specificity = len(pred_normal_metrics & true_normal_metrics) / len(true_normal_metrics) if len(true_normal_metrics) > 0 else 0.0
    bacc = (root_fault_recall + specificity) / 2

    return {
        "num_remained": len(pred_anomalous_metrics),
        "num_removed": len(pred_normal_metrics),
        "num_total": len(total_metrics),
        "reduction_rate": len(pred_normal_metrics) / len(total_metrics),
        "root_fault_recall": root_fault_recall,
        "root_fault_specificity": specificity,
        "root_fault_ba": bacc,
    }
