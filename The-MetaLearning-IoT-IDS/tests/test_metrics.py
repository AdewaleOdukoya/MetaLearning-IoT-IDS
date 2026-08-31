from src.evaluation.metrics import (
    compute_metrics,
    get_confusion_matrix,
    get_classification_report,
    print_metrics,
)

y_true = [0, 1, 2, 1, 0, 2]

y_pred = [0, 1, 1, 1, 0, 2]

metrics = compute_metrics(
    y_true,
    y_pred,
)

print_metrics(metrics)

print()

print(get_confusion_matrix(
    y_true,
    y_pred,
))

print()

report = get_classification_report(
    y_true,
    y_pred,
)

print(report.keys())