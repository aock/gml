# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import matplotlib.pyplot as plt


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
                if self.nodes[i][j] <= x[i] and x[i] <= self.nodes[i][j+1]:
                    phi[j] = 1 - ((x[i] - self.nodes[i][j]) / self.dist)
                    phi[j+1] = 1 - phi[j]
                    break
            out.append(phi)
        return np.hstack(out)

    """
    Performs the learning of the approximator
    @param X Matrix (Vector for 1D) containing all x-input values
    @param y Vector of corresponding output values
    """
    def learn(self, X, y):
        Xtilde = [self.transform(x) for x in X]
        Xdagger = np.linalg.pinv(Xtilde)
        #Xdagger = np.matmul(np.linalg.inv(np.matmul(np.transpose(Xtilde), Xtilde)), np.transpose(Xtilde)) 
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

    #np.random.shuffle(data)

    # Seperate data in input and output values
    dataX = [el[0:4] for el in data]
    dataY = [el[4] for el in data]

    # Normalize input data
    dataX = normalize(np.asarray(dataX))

    saveValues = 5000

    dataXTrain  = dataX[:saveValues]
    dataYTrain  = dataY[:saveValues]
    dataXTest   = dataX[saveValues:]
    dataYTest   = dataY[saveValues:]

    # Create hypothesis set
    for i in range(15, 25, 1):
        h = GLT(i, 4)
        # Learn on the training data
        h.learn(dataXTrain, dataYTrain)
        # Display the in-sample-error
        print('In-|Out-Sample-Error:  %d - %f | %f' % (i, calculateError(h, dataXTrain, dataYTrain), calculateError(h, dataXTest, dataYTest)))