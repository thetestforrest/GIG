#K-fold cross classify
import numpy as np
import pandas as pd
import math
import random

def Kfoldcrossclassify(sample, K, fun="cv3"):
    
    r = []
    if fun != "cv3":
        m = np.mat(sample)
        if fun == "cv1":
            t = 0
        else:
            t = 1
        mt = Kfoldcrossclassify(np.array(range(np.max(m[:, t]) + 1)), K)
        r = [[j for j in sample if j[t] in mt[i]] for i in range(K)]
        return r
    
    l = sample.shape[0]
    t = sample.copy()
    n = math.floor(l / K)
    retain = l - n * K
    
    for i in range(K - 1):
        nt = n
        e = len(t)
        # if e % n and e % K:
        if retain > i:
            nt += 1
        a = random.sample(range(e), nt)
        r.append([t[i] for i in a])
        t = [t[i] for i in range(e) if (i not in a)]
    r.append(t)
    
    return r