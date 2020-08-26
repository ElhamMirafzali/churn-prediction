from sklearn import metrics
import matplotlib.pyplot as plt


def draw_curves(targets, predictions, roc_save_path, pr_save_path):
    ##############################################################
    # calculate roc curve
    ##############################################################
    ns_probs = [0 for _ in range(len(targets))]
    ns_fpr, ns_tpr, _ = metrics.roc_curve(targets, ns_probs)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(targets, predictions)
    plt.figure()
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='b')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Proposed Model', color='r')
    plt.legend()
    plt.savefig(roc_save_path)
    plt.show()
    plt.close()

    ##############################################################
    # calculate precision-recall curve
    ##############################################################
    precision, recall, _ = metrics.precision_recall_curve(targets, predictions)
    # plot the precision-recall curves
    targets_count_one = targets.count(1.0)
    targets_len = len(targets)
    no_skill = targets_count_one / targets_len
    plt.figure()
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='b')
    plt.plot(recall, precision, marker='.', label='Proposed Model', color='r')
    plt.legend()
    plt.savefig(pr_save_path)
    plt.show()
    plt.close()
