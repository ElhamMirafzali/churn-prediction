from sklearn import metrics


def calculate_metrics(targets, predictions, bin_predictions):
    # calculate roc curve
    auc = metrics.roc_auc_score(targets, predictions)

    # calculate precision-recall curve
    precision_arr, recall_arr, _ = metrics.precision_recall_curve(targets, predictions)
    pr_auc = metrics.auc(recall_arr, precision_arr)

    # calculate precision
    average_precision = metrics.average_precision_score(targets, predictions)
    precision = metrics.precision_score(targets, bin_predictions)

    # calculate recall
    recall = metrics.recall_score(targets, bin_predictions)

    # calculate F1 score
    f1 = metrics.f1_score(targets, bin_predictions)

    return auc, pr_auc, average_precision, precision, recall, f1
