# operation lib

import numpy as np
from sklearn.metrics import roc_curve, auc

# 此函数用来计算准确率，计算预测对的占总数的百分比
def accuracy(preds, labels):
    # np.argmax(a, axis=0),找出每列最大的值，返回一个下标的元组，元组维数与列相同。
    # np.argmax(a, axis=1),找出每行最大的值，返回一个元组，元组维数与行相同。
    # np.sum() 在这里用来计算有多少对匹配的值
    # 所以此函数用来计算准确率，计算预测对的占总数的百分比
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1))
          / preds.shape[0])
# MSE是均方误差
def RMSE(p, y):
    N = p.shape[0]
    diff = p - y
    return np.sqrt((diff**2).mean())

def ROC_AUC(p, y):

    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc
