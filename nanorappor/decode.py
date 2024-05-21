"""
Decode RAPPOR responses
"""
from typing import List, Hashable
import numpy as np
import random
from collections import defaultdict
from sklearn.linear_model import Lasso
from response import RapporResponse
from bloom import BloomFilter

def get_bloom_estimates(responses: List[RapporResponse]):
    by_cohort = defaultdict(list) 

    for r in responses:
        by_cohort[r.cohort].append(r)
    
    cohorts = list(by_cohort.keys())
    m = len(cohorts)
    k = responses[0].k

    T = np.zeros((k, len(cohorts)))
    V = np.zeros((k, len(cohorts)))
    C = np.zeros_like(T)

    # cij = # of 1's set in cohort j
    for cohort, responses in by_cohort.items():
        i = cohorts.index(cohort)
        arr = np.array([r.report_array() for r in responses])
        C[:, i] = arr.sum(axis=0)

    
    N = [len(by_cohort[key]) for key in cohorts]
    N = np.array(N)

    ex = responses[0]
    f, p, q = ex.f, ex.p, ex.q

    p11 = q * (1 - f/2) + p * f/2
    p01 = p * (1 - f/2) + q * f/2
    p2 = p11 - p01
    
    for j in range(len(cohorts)):
        T[:, j] = (C[:, j] - ((p + 0.5 * f * q - 0.5 * f * p)) * N[j])/ ((1-f)*(q-p)) 
        ph = C[:, j] - p01*N[j] / (N[j] * p2)
        ph = np.clip(ph, 0, 1)
        r = ph * p11 + (1-ph) * p01
        V[:, j] = N[j] * r * (1-r) / (p2*p2)
        # T[:, j] /= C[:, j].sum()
    
    V = V.flatten('F')
    V = np.sqrt(V) # std dev

    return T.flatten('F'), V, cohorts

def get_design_matrix(h: int, k: int, cohorts: List[int], M: List[Hashable]):
    # make km x M 
    res = []
    for el in M:
        row = []
        for c in cohorts:
            random.seed(c)
            hash_seeds = [random.randint(0, 10000000) for _ in range(h)]
            bloom_filter = BloomFilter(hash_seeds, k)
            bloom_filter.add(el)
            row += bloom_filter.bitarr
        res.append(row)

    res = np.array(res)
    res = res.T
    assert res.shape == (k * len(cohorts), len(M)), "Design matrix wrong size"

    return res

def get_lasso_coefs(X: np.ndarray, y: np.ndarray):
    model = Lasso(alpha=1, positive=True)
    model.fit(X, y)

    return model.coef_

def resample_coefficients(Y, std):
    return Y + np.random.normal(0, std)

def decode_rappor_responses(responses: List[RapporResponse], 
                            M: List[Hashable],
                            thresh: float = 1.0):
    Y, std, cohorts = get_bloom_estimates(responses)
    h, k = responses[0].h, responses[0].k
    X = get_design_matrix(h, k, cohorts, M)
    #print(X, Y)
    all_coefs = []
    coefs = get_lasso_coefs(X, Y)
    all_coefs.append(coefs)

    for _ in range(4):
        Yprime = resample_coefficients(Y, std)
        coefs = get_lasso_coefs(X, Yprime)
        all_coefs.append(coefs)
    
    all_coefs = np.array(all_coefs)
    # get average coefs
    all_coefs_avg = all_coefs.mean(axis=0)
    all_coefs_std = all_coefs.std(axis=0)
    coef_err = all_coefs_std / np.sqrt(5)

    support,  = np.where(all_coefs_avg > 1e-6 + all_coefs_std*thresh)
    z = all_coefs_avg / coef_err
    p = np.exp(-0.717 * z - 0.416 * z * z)

    support = np.union1d(support, 
                         np.where(p < (0.05 / len(cohorts))))
    
    M = np.array(M)
    found = M[support]
    found_weights = all_coefs_avg[support]

    return found, found_weights, coef_err[support]
