import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(all_labels, all_preds, all_probs, n_classes):
    """ Compute classification metrics from ground-truth labels, predicted classes and predicted probabilities. """

    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    try:
        if n_classes == 2:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auroc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr', labels=list(range(n_classes)))
    except ValueError:
        # AUROC is undefined if a class is missing from the batch/fold
        auroc = float('nan')

    return {'accuracy': acc, 'balanced_accuracy': balanced_acc, 'macro_f1': macro_f1, 'auroc': auroc}
