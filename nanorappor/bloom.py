"""
Bloom filter
"""
import random
from typing import Hashable, List
from hashlib import md5

import time

BIGPRIME = 10**9 + 7

class RandomHashFunction:
    def __init__(self, n: int):
        random.seed(n)
        self.randmask = random.getrandbits(32)
        self.randmask2 = random.getrandbits(32)
        random.seed(time.time())
    def __call__(self, x: Hashable):
        hx = md5(str(x).encode()).hexdigest()
        hx = int(hx, 16)
        return ((hx * self.randmask) + self.randmask2) % BIGPRIME


class BloomFilter:
    
    def __init__(self,
                 hash_seeds: List[int],
                 array_length: int = 256):
        self.h = len(hash_seeds)
        self.k = array_length
        self.bitarr = [0 for _ in range(self.k)]
        self.seeds = hash_seeds
        self.hashes = [RandomHashFunction(x) for x in hash_seeds]

    def add(self, x):
        for idx in (f(x) % self.k for f in self.hashes):
            self.bitarr[idx] = 1

    def qry(self, x):
        for idx in (f(x) % self.k for f in self.hashes):
            if self.bitarr[idx] != 1:
                return False

        return True

