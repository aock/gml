from __future__ import division
import os, sys, numpy as np

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



if __name__ == "__main__":
    print("hello")