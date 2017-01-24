from __future__ import division
import numpy as np

from sklearn.neural_network import MLPRegressor
from OnlineLearning import generateConstant, generateComplex, generateSine, generateLinear, generateStep

import time
import pickle
import sys


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

    if sys.argv[1] == "load":
        with open(sys.argv[2], "rb") as f:
            e = pickle.load(f)
        print(e)


    k = eval(sys.argv[1])
    targetFunction = eval(sys.argv[2])
    saveName = "save_" + sys.argv[1] + "_" + sys.argv[2]

    print k, targetFunction, saveName
    error = np.zeros([200, 200])

    print("Starting")
    for i in range(1,201):
        t = time.time()
        for j, r in enumerate(frange(0, 1, 0.05)):
            X, Y = targetFunction(n=i, noise=r)
            if len(X) == 1:
                X = X.reshape(1, -1)
            else:
                X = X.reshape(-1, 1)
            a = MLPRegressor(solver='sgd', activation ='tanh',
                       hidden_layer_sizes=[5]*k, max_iter=int(1e6), tol=1e-8)
            m = a.fit(X, Y)
            error[i-1, j] = calculategtLoss(targetFunction, m)
        print(i, "finished in", time.time() - t, "seconds | error: ")

    with open(saveName, "wb") as f:
        pickle.dump(error, f)


