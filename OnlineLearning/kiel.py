from __future__ import division
import os, sys, numpy as np
import matplotlib.pyplot as plt
from OLLib import *
"""
# 2008
mo8  = np.array([6+i*7 for i in range(53)])
dio8 = np.array([[i*7, 1+i*7, 2+i*7] for i in range(53)]).flatten()
fr8  = np.array([3+i*7 for i in range(53)])
sa8  = np.array([4+i*7 for i in range(53)])
ff8  = np.array([1,80,83,121,132,142,227,276,305,359,360])
sof8  = np.concatenate((np.array([5+i*7 for i in range(53)]), ff8))
np.sort(sof8)

mo8 = mo8[np.where(mo8 < 365)]
dio8 = dio8[np.where(dio8 < 365)]
fr8 = fr8[np.where(fr8 < 365)]
sa8 = sa8[np.where(sa8 < 365)]
sof8 = sof8[np.where(sof8 < 365)]



print(mo8)
print(dio8)
print(fr8)
print(sa8)
print(sof8)
"""
def batch(data, approx):
    Xtilde = [approx.transform(x) for x in data[:,1]]
    Xdagger = np.linalg.pinv(Xtilde)
#    data[:,2] += 1
    approx.w = np.dot(Xdagger, data[:,2])

def learn(data, approximator, learner):
    for i, x in enumerate(data):
        learner.learn(x[1], x[2])


if __name__ == "__main__":
    np.random.seed(12345)
    first = np.fromfile('firstYear.dat', sep=' ').reshape([35040, 3])
    approx = GLT(200, -10, 10)
    batch(first, approx)
#    second = np.fromfile('secondYear.dat', sep=' ').reshape([35040, 3])
#    learner = RLSLearner(approx)
#    learn(second, approx, learner)
    predict = []
    for m in range(96):
        predict.append(approx.evaluate(m/96-10))
    plt.plot(range(96), first[:96,2])
    plt.show()
