import numpy as np
import pandas as pd

def max_ks(predict_prob, DV):
    merged = np.dstack((DV, predict_prob))
    merged = merged[0]
    mergesort = merged[np.argsort(merged[:,0])]
    total_bad = np.unique(mergesort[:,1], return_counts=True)[1][0]
    total_good =  np.unique(mergesort[:,1], return_counts=True)[1][1]
    count0 =0
    count1 = 0
    max_ks =0
    for i,j in mergesort:
        if j == 1:
            count1 += 1
        elif j==0:
            count0 += 1
        ks = abs(count1 / float(total_good) -  count0/float(total_bad)) * 100
        if ks > max_ks:
            max_ks = ks
    return max_ks

def ks_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return max_ks(y.values if hasattr(y, "values") else y, y_pred_proba)

def calculate_iv(df, feature, target, bins=10):
    df = df[[feature, target]].copy()
    
    # Bin feature
    df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
    
    grouped = df.groupby("bin")[target].agg(["count", "sum"])
    grouped.columns = ["total", "bads"]
    
    grouped["goods"] = grouped["total"] - grouped["bads"]
    
    # Distribution
    grouped["dist_bad"] = grouped["bads"] / grouped["bads"].sum()
    grouped["dist_good"] = grouped["goods"] / grouped["goods"].sum()
    
    # Avoid division by zero
    grouped["woe"] = np.log(
        (grouped["dist_good"] + 1e-6) /
        (grouped["dist_bad"] + 1e-6)
    )
    
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    
    return grouped["iv"].sum()
