# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import matplotlib.pyplot as plt



class RLS:
    def __init__(self, dim, forgF):
        # covariance matrix
        self.P = np.identity(dim)
        # forgetting factor
        self.r = forgF

        self.bestW = None

    def learn(self, x, y, yh):
        w = np.dot(self.P, x) * (y - yh) / (self.r + np.inner(x, np.dot(self.P, x)))
        self.P = self.P / self.r - np.outer(np.dot(self.P, x), np.dot(x, self.P)) \
                                   / (self.r * (self.r + np.inner(np.dot(x, self.P), x)))

        return w

