from sklearn import metrics
from matplotlib import pyplot


def draw_curves(targets, predictions):
    ##############################################################
    # calculate roc curve
    ##############################################################
    ns_probs = [0 for _ in range(len(targets))]
    ns_fpr, ns_tpr, _ = metrics.roc_curve(targets, ns_probs)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(targets, predictions)
    plt1 = pyplot.figure(1)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Proposed Model')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()

    ##############################################################
    # calculate precision-recall curve
    ##############################################################
    precision, recall, _ = metrics.precision_recall_curve(targets, predictions)
    # plot the precision-recall curves
    targets_count_one = targets.count(1.0)
    targets_len = len(targets)
    no_skill = targets_count_one / targets_len
    plt2 = pyplot.figure(2)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='Proposed Model')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()

    pyplot.show()
