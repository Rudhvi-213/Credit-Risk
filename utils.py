import numpy as np

def max_ks(DV, predict_prob):
    merged = np.dstack((predict_prob, DV))
    merged = merged[0]
    mergesort = merged[np.argsort(merged[:,0])]
    total_bad = np.unique(mergesort[:,1], return_counts=True)[1][0]
    total_good = np.unique(mergesort[:,1], return_counts=True)[1][1]
    count0 = 0
    count1 = 0
    max_ks = 0
    for i,j in mergesort:
        if j==1:
            count1 += 1
        elif j==0:
            count0 += 1
        ks = abs(count1 / float(total_good) - count0/float(total_bad))*100

        if ks > max_ks:
            max_ks = ks
    return max_ks

def ks_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return max_ks(y.values if hasattr(y, "values") else y, y_pred_proba)