# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np

"""
Calcuate the MSE on a given dataset
@param h The hypothesis to evaluate
@param x The input values of the test data
@param y The output value of the test data
@return The mean squared error
"""
def calculateError(h, x, y):
    error = 0
    for el in zip(x,y):
        yh = h.evaluate(el[0])
        y = el[1]
        error += (yh-y)**2
    error /= len(x)
    return error

"""
Very simple approximator performing the linear regression without transformation
"""        
class NaiveApproximator():
    def __init__(self, dim):
        self.w = np.zeros(dim)
    
    def transform(self, x):
        return x
    
    def learn(self, X, y):
        Xdagger = np.linalg.pinv(X)
        self.w = np.dot(Xdagger, y)
    
    def evaluate(self, x):
        return np.dot(self.w, self.transform(x))
    
if __name__ == "__main__":
    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
    # Seperate data in input and output values
    dataX = [el[0:4] for el in data]
    dataY = [el[4] for el in data]
    # Create hypothesis set
    h = NaiveApproximator(4)
    # Learn on the training data
    h.learn(dataX, dataY)
    # Display the in-sample-error
    print('In-Sample-Error:', calculateError(h, dataX, dataY))