# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:25:24 2016

@author: Jonas Schneider

Template for the CCPP-Challenge

"""

import numpy as np
import matplotlib.pyplot as plt

flag = True

class Polynomial():
    """
    Constructor
    @param deg The degree of the polynomial
    """

    def __init__(self, deg, ):
        self.deg = deg
        self.w = None

    """
    Tansformation function (phi)
    @param x 1D-input value to transform
    @return phi(x)
    """

    def transform(self, x):
        global flag
        func = [1]
        gen = [x_dim ** i for x_dim in x for i in range(1, self.deg + 1)]
        for g in gen:
            func.append(g)
        for x_ in x:
            func.append(np.sin(x_))
        if flag:
            print(func)
            flag = False
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
