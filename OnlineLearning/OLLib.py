# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:38:37 2017

@author: Jonas Schneider
@author: Matthias Greshake
"""

import numpy as np


class GLT:
    """
    This class implements a grid-based lookup table approximator for 1D-Regression
    """
    def __init__(self, deg, xmin=-1, xmax=1, pos=None):
        """
        Constructor
        @param deg Number of nodes
        @param xmin Minimum border of the input space
        @param xmax Maximum border of the input space
        @param pos Position of the nodes. If left None the nodes are placed equidistantly in the input space
        """
        self.deg = deg
        self.w = np.zeros(deg)
        self.nodes = np.linspace(xmin, xmax, deg)
        self.dist = np.diff(self.nodes)

    def transform(self, x):
        """
        Non-linear transformation function phi(x) for the GLT
        @param x Input coordinate to transform
        @return The transformed vector phi(x)
        """
        phi = np.zeros(self.deg)
        for i, n in enumerate(self.nodes):
            if self.nodes[i] <= x <= self.nodes[i+1]:
                phi[i] = 1 - ((x - self.nodes[i]) / self.dist[i])
                phi[i+1] = 1 - phi[i]
                return phi

    def evaluate(self, x):
        """
        Evaluation of the approximator at a given position in the input space
        @param x Coordinate where the approximator shall be evaluated
        @return Evaluation of the approximator phi(x) * w
        """
        return np.dot(self.w, self.transform(x))


class Polynomial:
    """
    This class implements a polynomial approximator for 1D-Regression
    """
    def __init__(self, deg):
        """
        Constructor
        @param deg Degree of the polynomial
        """
        self.deg = deg
        self.w = np.zeros(deg + 1)

    def transform(self, x):
        """
        Non-linear transformation function phi(x) for the polynomial
        @param x Input coordinate to transform
        @return The transformed vector phi(x)
        """
        return np.array([x**i for i in range(self.deg + 1)])

    def evaluate(self, x):
        """
        Evaluation of the approximator at a given position in the input space
        @param x Coordinate where the approximator shall be evaluated
        @return Evaluation of the approximator phi(x) * w
        """
        return np.dot(self.w, self.transform(x))


class PALearner:
    """
    This class implements the Passive-Aggressive learning algorithm
    """
    def __init__(self, approximator, param=None):
        """
        Constructor
        @param approximator The approximator the learning algorithm performs on (e.g. instance of GLT/Polynomial)
        @param param Empty
        """
        self.approx = approximator

    def learn(self, x, y):
        """
        Performs a PA learning step on the approximator associated with the learner
        The parameter vector is udpated directly in the approximator object
        @param x The input coordinate where to learn
        @param y The value to learn
        """
        phiX = self.approx.transform(x)
        yp = self.approx.evaluate(x)
        self.approx.w += phiX * (y - yp) / (1 + np.inner(phiX, phiX))


class RLSLearner:
    """
    This class implements the Recursive Least Squares learning algorithm
    """
    def __init__(self, approximator, param={"pInit":100000, "l":1.0}):
        """
        Constructor
        @param approximator The approximator the learning algorithm performs on (e.g. instance of GLT/Polynomial)
        @param param Dicitionary containing the RLS parameters:
                        pInit Initialization value for the covariance matrix main diagonal
                        l Forgetting factor lambda
        """
        self.approx = approximator
        self.P = np.identity(len(approximator.w)) * param["pInit"]
        self.l = param["l"]

    def learn(self, x, y):
        """
        Performs a RLS learning step on the approximator associated with the learner
        The parameter vector is updated directly in the approximator object
        @param x The input coordinate where to learn
        @param y The value to learn
        """
        phiX = self.approx.transform(x)
        yp = self.approx.evaluate(x)
        self.approx.w += np.dot(self.P, phiX) * (y - yp) / (self.l + np.inner(phiX, np.dot(self.P, phiX)))
        self.P = self.P / self.l - np.outer(np.dot(self.P, phiX), np.dot(phiX, self.P)) \
                                   / (self.l * (self.l + np.inner(np.dot(phiX, self.P), phiX)))
