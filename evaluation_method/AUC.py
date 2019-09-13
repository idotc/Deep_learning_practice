# encoding: utf-8
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def compute_AUC(prob, labels):
    samples = zip(prob, labels)
    rank = [v2 for v1, v2 in sorted(samples, key = lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i] == 1]
    posCount = sum(labels)
    negCount = len(labels) - posCount
    AUC = (sum(rankList) - posCount*(posCount+1)/2)/(posCount*negCount)
    return AUC

labels = [0, 0, 1, 1]
prob = [0.1, 0.4, 0.35, 0.8]
print ("Ours cal.auc:", compute_AUC(prob, labels))

labels = [1, 0, 0, 0, 1, 0, 1, 0]
prob = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
print ("Ours cal.auc:", compute_AUC(prob, labels))

fpr, tpr, thresholds = roc_curve(labels, prob, pos_label = 1)
print ("sklean cal.auc:", auc(fpr, tpr))
