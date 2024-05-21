"""
Rappor response algorithm
"""
from typing import Hashable, List
import random
import time
from bloom import BloomFilter

class RapporResponse:
    def __init__(self,
                 v: Hashable,
                 cohort: int,
                 h: int,
                 k: int,
                 f: float,
                 p: float,
                 q: float):

        self.k = k
        self.h = h
        self.f = f
        self.p = p
        self.q = q
        self.cohort = cohort

        # 1. Hash v onto Bloom filter w/ hash_seeds, size k
        random.seed(cohort)
        hash_seeds = [random.randint(0, 10000000) for _ in range(h)]
        self.bloom_filter = BloomFilter(hash_seeds, k)
        self.bloom_filter.add(v)

        # 2. Permanent randomized response 
        self.permanent = []
        
        # Unseed for the permanent randomized response.
        random.seed(time.time())
        for i in range(k):
            r = random.random()

            if r < 0.5 * f:
                self.permanent += [1]
            elif r < f:
                self.permanent += [0]
            else:
                self.permanent += [self.bloom_filter.bitarr[i]]
    
    def report_array(self):
        report_arr = []
        for i in range(self.k):
            prob = self.q if self.permanent[i] == 1 else self.p
            if random.random() < prob:
                report_arr += [1]
            else:
                report_arr +=[0]

        return report_arr

