# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:38:37 2017

@author: Jonas Schneider
"""

import numpy as np

"""
This class implements a grid-based lookup table approximator for 1D-Regression
"""
class GLT:
    """
    Constructor
    @param deg Number of nodes
    @param xmin Minimum border of the input space
    @param xmax Maximum border of the input space
    @pos Position of the nodes. If left None the nodes are placed equidistantly in the input space
    """
    def __init__(self, deg, xmin=-1, xmax=1, pos=None):
        return 0
    
    """
    Non-linear transformation function phi(x) for the GLT
    @param x Input coordinate to transform
    @return The transformed vector phi(x)
    """
    def transform(self, x):
        return 0

    """
    Evaluation of the approximator at a given position in the input space
    @param x Coordinate where the approximator shall be evaluated
    @return Evaluation of the approximator phi(x)*w
    """
    def evaluate(self, x):
        return 0
        
"""
This class implements a polynomial approximator for 1D-Regression
"""
class Polynomial:
    """
    Constructor
    @param deg Degree of the polynomial
    """
    def __init__(self, deg):
        return 0
        
    """
    Non-linear transformation function phi(x) for the GLT
    @param x Input coordinate to transform
    @return The transformed vector phi(x)
    """        
    def transform(self, x):
        return 0
        
    """
    Evaluation of the approximator at a given position in the input space
    @param x Coordinate where the approximator shall be evaluated
    @return Evaluation of the approximator phi(x)*w
    """
    def evaluate(self, x):
        return 0

"""
This class implements the Passive-Aggressive learning algorithm
"""
class PALearner:
    """
    Constructor
    @param approximator The approximator the learning algorithm performs on (e.g. instance of GLT/Polynomial)
    @param param Empty
    """
    def __init__(self, approximator, param=None):
        return 0
       
    """
    Performs a PA learning step on the approximator associated with the learner
    The parameter vector is udpated directly in the approximator object
    @param x The input coordinate where to learn
    @param y The value to learn
    """
    def learn(self, x, y):
        return 0
        
"""
This class implements the Passive-Aggressive learning algorithm
"""
class RLSLearner:
    """
    Constructor
    @param approximator The approximator the learning algorithm performs on (e.g. instance of GLT/Polynomial)
    @param param Dicitionary containing the RLS parameters:
                    pInit Initialization value for the P-Matrix main diagonal
                        l Forgetting-Factor Lambda
    """
    def __init__(self, approximator, param={"l":1.0, "pInit":100000}):
        return 0
        
    """
    Performs a RLS learning step on the approximator associated with the learner
    The parameter vector is udpated directly in the approximator object
    @param x The input coordinate where to learn
    @param y The value to learn
    """    
    def learn(self, x, y):
        return 0
        


