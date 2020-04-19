from sklearn import metrics


def calculate_metrics(targets, predictions, bin_predictions):
    # calculate roc curve
    auc = metrics.roc_auc_score(targets, predictions)

    # calculate precision-recall curve
    precision, recall, _ = metrics.precision_recall_curve(targets, predictions)
    pr_auc = metrics.auc(recall, precision)

    # calculate precision
    average_precision = metrics.average_precision_score(targets, predictions)

    # calculate recall
    average_recall = metrics.recall_score(targets, bin_predictions)

    return auc, pr_auc, average_precision, average_recall
