from __future__ import division
import numpy as np

from sklearn.neural_network import MLPRegressor
from OnlineLearning import generateConstant

import time


def frange(min, max, step):
    tmp = min
    while tmp <= max:
        yield tmp
        tmp += step

def calculategtLoss(targetFunc, approximator, funcParam=None, numPoints=1000):
    gtX = np.linspace(-1, 1, numPoints)
    gtY = targetFunc(gtX, param=funcParam, noise=0)
    evalY = np.zeros(numPoints)
    for i, x in enumerate(gtX):
        evalY[i] = approximator.predict(x)
    return 1 / numPoints * np.sum(np.power(np.subtract(gtY, evalY), 2))

if __name__ == "__main__":

    print("Generating Regressors")

    clf_5 =   MLPRegressor(solver='sgd', activation ='tanh',
                           hidden_layer_sizes=[5], max_iter=int(1e6), tol=1e-8)
    clf_55 =  MLPRegressor(solver='sgd', activation ='tanh',
                           hidden_layer_sizes=[5,5], max_iter=int(1e6), tol=1e-8)
    clf_555 = MLPRegressor(solver='sgd', activation ='tanh',
                           hidden_layer_sizes=[5,5,5], max_iter=int(1e6), tol=1e-8)


    targetFunction = generateConstant
    error = np.zeros([3, 200, 200])

    print("Starting")
    for i in range(1,201):
        t = time.time()
        for j, r in enumerate(frange(0, 1, 0.05)):
            X, Y = targetFunction(n=i, noise=r)
            if len(X) == 1:
                X = X.reshape(1, -1)
            else:
                X = X.reshape(-1, 1)
            for k in range(1,4):#, clf_55, clf_555]):
                a = MLPRegressor(solver='sgd', activation ='tanh',
                           hidden_layer_sizes=[5]*k, max_iter=int(1e6), tol=1e-8)
                m = a.fit(X, Y)
                error[k-1, i-1, j] = calculategtLoss(targetFunction, m)
        print(i, "finished in", time.time() - t, "seconds | error: ")




