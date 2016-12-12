# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import matplotlib.pyplot as plt


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
        # Xdagger = np.matmul(np.linalg.inv(np.matmul(np.transpose(Xtilde), Xtilde)), np.transpose(Xtilde))
        self.w = np.dot(Xdagger, y)

    """
    Evaluate the polynomial approximator
    @param x Point to evaluate
    @return Predicted value
    """

    def evaluate(self, x):
        return np.dot(self.w, self.transform(x))


if __name__ == "__main__":
    # Read data from file
    data = np.genfromtxt("ccpp_first_batch.txt", delimiter=",")
    # Seperate data in input and output values

    for el in data:
        print(el)

        break

    dataX0 = [np.array(el)[0] for el in data]
    print(dataX0)

    # dataX = [list( np.array(el)[[0,1,2,3]]) for el in data]
    # dataX0 = [list( np.array(el)[[0]]) for el in data]
    dataY = [el[4] for el in data]
    # Normalize input data
    # dataX = normalize(np.asarray(dataX0))
    print("Data read and normalized")

    for i in range(4, 10):
        p = Polynomial(i)
        p.learn(dataX0, dataY)

        print("In-Sample-Error for P(%d): %d" % (i, calculateError(p, dataX0, dataY)))

    import sys

    sys.exit()

    # Create hypothesis set
    h = GLT(100, len(dataX[0]))
    # Initialize learning algorithm
    l = RLS(100 * len(dataX[0]), 1)
    # Learn on the training data
    h.learn(dataX, dataY, l)
    # Display the in-sample-error
    print('In-Sample-Error:', calculateError(h, dataX, dataY))
    # plot(dataX, [h.evaluate(i)[0] for i in dataX], dataY)