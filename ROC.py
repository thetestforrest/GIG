import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from GIPK import getSimilarMatrix
from Kfold import Kfoldcrossclassify
from GiGs import GiGs

def ROC_compute(virusdrug, drugsim, virussim, a, b):
    
    note_auc = []
    note_aupr = []

    for h in range(20):
        f1 = Kfoldcrossclassify(a, 5, fun='cv3')
        f2 = Kfoldcrossclassify(b, 5, fun='cv3')
        sum_auc, ACC, SEN, SPE = (0, 0, 0, 0)
        for m in range(5):
            test_sample = np.array(f1[m])
            negative_sample = np.array(f2[m])
            virusdrug_ = virusdrug.copy()
            virusdrug_[test_sample[:, 0], test_sample[:, 1]] = 0
            
            #best parameters
            VDA = GiGs(virusdrug_, drugsim, virussim, 70, 1, 16, 0.125, 16)
            test_sample_number = test_sample.shape[0]
            negative_sample_number = negative_sample.shape[0]
            label = test_sample_number*[1]+negative_sample_number*[0]
            label = np.array(label)
            sample = np.vstack((test_sample, negative_sample))
            score = VDA[sample[:, 0], sample[:, 1]]
            fpr, tpr, threshold = roc_curve(label,score)
            pre, rec, threshold = precision_recall_curve(label,score)
            
            sumACC = 0
            sumSEN = 0
            sumSPE = 0
            for j in range(threshold.size):
                TP, TN, FP, FN = (0, 0, 0, 0)
                threshold_value = threshold[j]
                for k in range(score.size):
                    predicted_value = score[k]
                    if predicted_value >= threshold_value:
                        if label[k]:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if label[k]:
                            FN += 1
                        else:
                            TN += 1
                sumACC += (TP+TN)/(TP+TN+FP+FN)
                sumSEN += TP/(TP+FN)
                sumSPE += TN/(TN+FP)
            
            ACC += sumACC / (threshold.size)
            SEN += sumSEN / (threshold.size)
            SPE += sumSPE / (threshold.size)
            auc_pre = auc(fpr, tpr)
            aupr_pre = auc(rec, pre)
            sum_auc += auc_pre
            note_auc.append((auc_pre, fpr, tpr))
            note_aupr.append((aupr_pre, rec, pre))
    
    return note_auc,note_aupr
