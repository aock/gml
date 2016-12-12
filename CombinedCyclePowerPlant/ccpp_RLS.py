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

    def learn(self, x, y, yh):
        w = np.dot(self.P, x) * (y - yh) / (self.r + np.inner(x, np.dot(self.P, x)))
        self.P = self.P / self.r - np.outer(np.dot(self.P, x), np.dot(x, self.P)) \
                                   / (self.r * (self.r + np.inner(np.dot(x, self.P), x)))
        return w

    
if __name__ == "__main__":
    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
    # Seperate data in input and output values

    dataX = [list( np.array(el)[[0,1,2,3]]) for el in data]
    dataY = [el[4] for el in data]
    # Normalize input data
    dataX = normalize(np.asarray(dataX))
    # Create hypothesis set
    h = GLT(100, len(dataX[0]))
    # Initialize learning algorithm
    l = RLS(100 * len(dataX[0]), 1)
    # Learn on the training data
    h.learn(dataX, dataY, l)
    # Display the in-sample-error
    print('In-Sample-Error:', calculateError(h, dataX, dataY))
    #plot(dataX, [h.evaluate(i)[0] for i in dataX], dataY)