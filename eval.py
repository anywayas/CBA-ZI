# -*- coding: utf-8 -*-
from sklearn import metrics
import numpy as np


def one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

# def multi_class_eval(all_probs_pred, target):
#     pred = all_probs_pred.argmax(-1)
#     acc = metrics.accuracy_score(target, pred)
#     micro_f1_score = metrics.f1_score(target, pred, average="micro")
#     macro_f1_score = metrics.f1_score(target, pred, average='macro')
#     micro_precision = metrics.precision_score(target, pred, average="micro")
#     macro_precision = metrics.precision_score(target, pred, average='macro')
#     micro_recall = metrics.recall_score(target, pred, average="micro")
#     macro_recall = metrics.recall_score(target, pred, average='macro')
#     try:
#         y_test = one_hot(np.array(target), 86)
#         predic = one_hot(np.array(pred), 86)
#         precision, recall, thresholds = metrics.precision_recall_curve(y_test.ravel(), predic.ravel())
#         micro_auprc = metrics.auc(precision, recall)
#     except Exception:
#         micro_auprc = 0
#
#     return acc, micro_f1_score, macro_f1_score, micro_precision,macro_precision,micro_recall,macro_recall,micro_auprc


def real_fake_eval(real_probs_pred, target):
    """do_compute_metrics
    real_probs_pred:[num,]
    target:[num,]
    """
    #compute optimal_threshold
    pre, rec, threshold_list = metrics.precision_recall_curve(target, real_probs_pred)
    f1 = (2*pre*rec)/(pre+rec)
    optimal_f1_index = np.argmax(f1)
    optimal_threshold = threshold_list[optimal_f1_index]
    pred = (real_probs_pred >= optimal_threshold).astype(np.long)
    optimal_acc = metrics.accuracy_score(target, pred)

    f1_score = metrics.f1_score(target, pred)
    auc_roc = metrics.roc_auc_score(target, real_probs_pred)
    auc_prc = metrics.auc(rec, pre)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    norm_pred = (real_probs_pred >= 0.5).astype(np.long)
    acc = metrics.accuracy_score(target, norm_pred)

    return acc, auc_roc, auc_prc, precision, recall, f1_score, optimal_acc


# if __name__ == '__main__':
#     a = np.asarray([0.1, 0.2, 0.5, 0.9,0.8,0.8])
#     b = np.asarray([0, 1, 1, 1,1,1])
#     real_fake_eval(a,b)

    # a = np.asarray([[0.1,0.1,0.5,0.9], [0.8,0.3,0.6,0.9], [0.8,0.6,0.5, 0.2]])
    # b = np.asarray([1, 3, 2])
    # acc, micro_f1_score, macro_f1_score, micro_precision,macro_precision,micro_recall,macro_recall,micro_auprc = multi_class_eval(a, b)
    # print("acc:{:.4f}, micro_f1_score:{:.4f},macro_f1_score:{:.4f}, micro_precision:{:.4f}, macro_precision:{:.4f}, "
    #       "micro_recall:{:.4f},macro_recall:{:.4f},micro_auprc:{:.4f}" \
    #       .format(acc, micro_f1_score, macro_f1_score, micro_precision, macro_precision, micro_recall, macro_recall,
    #               micro_auprc))