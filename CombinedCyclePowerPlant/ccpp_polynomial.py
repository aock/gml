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
    for el in zip(x, y):
        yh = h.evaluate(el[0])
        y = el[1]
        error += (yh - y)**2
    error /= len(x)
    return error


def normalize(x):
    for i in range(len(x[0])):
        x[:,i] -= np.amin(x[:,i])
        x[:,i] /= np.amax(x[:,i])
    return x


"""
This class contains a linear regressor on a polynomial-approximator
for 1D problems
"""
class Polynomial():
    """
    Constructor
    @param deg The degree of the polynomial
    """

    def __init__(self, deg):
        self.deg = deg
        self.w = np.zeros(deg + 1)

    """
    Tansformation function (phi)
    @param x 1D-input value to transform
    @return phi(x)
    """

    def transform(self, x):
        func = [x_dim ** i for x_dim in x for i in range(self.deg + 1) ]
        for x_ in x:
            func.append(np.sin(x_))
        return func

    """
    Performs the learning of the approximator
    @param X Matrix (Vector for 1D) containing all x-input values
    @param y Vector of corresponding output values
    """

    def learn(self, X, y):
        Xtilde = [self.transform(x) for x in X]
        Xdagger = np.linalg.pinv(np.array(Xtilde))
        self.w = np.dot(Xdagger, y)

    """
    Evaluate the polynomial approximator
    @param x Point to evaluate
    @return Predicted value
    """
    def evaluate(self, x):
        Xdagger = self.transform(x)
        return np.dot(self.w, Xdagger), Xdagger

    
if __name__ == "__main__":
    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
    # Shuffle data
    np.random.shuffle(data)
    # Seperate data in input and output values
    dataX = [el[0:4] for el in data]
    dataY = [el[4] for el in data]
    # Normalize input data
    dataX = normalize(np.asarray(dataX))
    # Split data in training and test set
    trainX = dataX[:6000]
    trainY = dataY[:6000]
    testX = dataX[6000:]
    testY = dataY[6000:]
    # Create hypothesis set
    h = Polynomial(10)
    # Learn on the training data
    h.learn(trainX, trainY)
    # Display the in-sample-error
    print('In-Sample-Error:', calculateError(h, testX, testY))
