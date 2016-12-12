# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import matplotlib.pyplot as plt

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
        error += (yh - y) ** 2
    error /= len(x)
    return error


def normalize(x):
    for i in range(len(x[0])):
        # x[:,i] -= np.amin(x[:,i])
        x[:, i] /= np.amax(x[:, i])
    return x


def plot(x, y1, y2):
    plt.plot(x, y2, 'bo', x, y1, 'ro')
    plt.show()


"""
This class contains a linear regressor on a GLT-approximator
for 1D problems
"""


class GLT:
    """
    Constructor
    The nodes are implicity placed equidistantly on [-1;1]
    For a general solution neither the input interval of [-1;1] nor the
    equidistant positioning should be assumed
    @param deg Number of nodes
    """

    def __init__(self, deg, dim):
        self.deg = deg
        self.w = np.zeros(deg * dim)
        self.nodes = [np.linspace(0, 1, deg) for i in range(dim)]
        self.dist = self.nodes[0][1] - self.nodes[0][0]

    """
    Tansformation function (phi)
    @param x 1D-input value to transform
    @return phi(x)
    """

    def transform(self, x):
        out = []
        for i in range(len(x)):
            phi = np.zeros(self.deg)
            for j in range(self.deg - 1):
                if self.nodes[i][j] <= x[i] and x[i] <= self.nodes[i][j + 1]:
                    phi[j] = 1 - ((x[i] - self.nodes[i][j]) / self.dist)
                    phi[j + 1] = 1 - phi[j]
                    break
            out.append(phi)
        return np.hstack(out)

    """
    Performs the learning of the approximator
    @param X Matrix (Vector for 1D) containing all x-input values
    @param y Vector of corresponding output values
    """

    def learn(self, X, y, l):
        for i, x in enumerate(X):
            yh, Xdagger = self.evaluate(x)
            self.w += l.learn(Xdagger, y[i], yh)

    """
    Evaluate the polynomial approximator
    @param x Point to evaluate
    @return Predicted value
    """

    def evaluate(self, x):
        Xdagger = self.transform(x)
        return np.dot(self.w, Xdagger), Xdagger


class RLS:
    def __init__(self, dim, forgF):
        # covariance matrix
        self.P = np.identity(dim)
        # forgetting factor
        self.r = forgF

    def learn(self, x, y, yh):
        w = np.dot(self.P, x) * (y - yh) / (self.r + np.inner(x, np.dot(self.P, x)))
        self.P = self.P / self.r - np.outer(np.dot(self.P, x), np.dot(x, self.P)) / (
        self.r * (self.r + np.inner(np.dot(x, self.P), x)))
        return w


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