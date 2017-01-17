from __future__ import division
import os, sys, numpy as np
import matplotlib.pyplot as plt
from OLLib import *

# 2008
dio8 = np.array([[i*7, 1+i*7, 2+i*7] for i in range(53)]).flatten()
fr8  = np.array([3+i*7 for i in range(53)])
sa8  = np.array([4+i*7 for i in range(53)])
so8 = np.array([5+i*7 for i in range(53)])
mo8  = np.array([6+i*7 for i in range(53)])
ff8  = np.array([1,81,84,122,133,143,228,277,306,360,361])
sof8 = np.concatenate((ff8,so8))
np.sort(sof8)

mo8 = np.array([x for x in mo8 if x not in ff8])
dio8 = np.array([x for x in dio8 if x not in ff8])
fr8 = np.array([x for x in fr8 if x not in ff8])
sa8 = np.array([x for x in sa8 if x not in ff8])

for arr in [mo8, dio8, fr8, sa8, sof8]:
    for i,e in enumerate(arr):
        if e > 59:
            arr[i] -= 1

mo8 = mo8[np.where(mo8 < 365)]
dio8 = dio8[np.where(dio8 < 365)]
fr8 = fr8[np.where(fr8 < 365)]
sa8 = sa8[np.where(sa8 < 365)]
sof8 = sof8[np.where(sof8 < 365)]

# 2009
dio9 = np.array([[i*7, 5+i*7, 6+i*7] for i in range(53)]).flatten()
fr9  = np.array([1+i*7 for i in range(53)])
sa9  = np.array([2+i*7 for i in range(53)])
so9 = np.array([3+i*7 for i in range(53)])
mo9  = np.array([4+i*7 for i in range(53)])
ff9  = np.array([1,6,100,103,121,141,152,162,227,276,359,360])
sof9 = np.concatenate((ff9,so9))
np.sort(sof9)

mo9 = np.array([x for x in mo9 if x not in mo9])
dio9 = np.array([x for x in dio9 if x not in dio9])
fr9 = np.array([x for x in fr9 if x not in fr9])
sa9 = np.array([x for x in sa9 if x not in sa9])

mo9 = mo9[np.where(mo9 < 365)]
dio9 = dio9[np.where(dio9 < 365)]
fr9 = fr9[np.where(fr9 < 365)]
sa9 = sa9[np.where(sa9 < 365)]
sof9 = sof9[np.where(sof9 < 365)]

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
#    batch(first, approx)
#    second = np.fromfile('secondYear.dat', sep=' ').reshape([35040, 3])
#    learner = RLSLearner(approx)
#    learn(second, approx, learner)
    predict = []
    for m in range(96):
        predict.append(approx.evaluate(m/96-10))
    plt.plot(range(96), first[:96,2])
    plt.show()
