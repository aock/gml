import numpy as np
from copy import deepcopy as dc
from time import time

from featureExtraction import FeatureExtraction

from sklearn.svm import SVC
import sys


def getFeatures(d, fe):
    features = []
    result = []
    for l in data2:
        pic = np.reshape(l[1:], [28, 28]).astype(int)
        f = fe.get_train_data(pic)
        features.append(dc(f))
        result.append(int(l[0]))
    return features, result


if __name__ == "__main__":

    batchName = "mnist_first_batch.csv"
    batchName2 = "mnist_first_test.csv"
    testName = "mnist_second_test.csv"

    fe = FeatureExtraction()

    print("Load data ...")
    data = np.genfromtxt(batchName, delimiter=',')
    data2 = np.genfromtxt(batchName2, delimiter=',')
    test = np.genfromtxt(testName, delimiter=',')

    print("Start FE ...", end='')
    sys.stdout.flush()

    t_s = time()

    features, result = getFeatures(data, fe)
    features2, result2 = getFeatures(data, fe)
    features = np.concatenate((np.array(features), np.array(features2)))
    result = np.concatenate((np.array(result), np.array(result2)))
    testX, testY = getFeatures(data2, fe)
    print(" finished after", time() - t_s, "seconds")
    t_c = time()

    print("Fitting ...", end='')
    sys.stdout.flush()
    svm = SVC(C=2, kernel='rbf', gamma='auto')
    svm.fit(features, result)
    print(" finished after", time() - t_c, "seconds")
    t_c = time()

    print("Predicting ...", end='')
    sys.stdout.flush()
    resY = svm.predict(testX)
    print(" finished after", time() - t_c, "seconds")

    a = len(resY[resY != testY])
    b = len(resY)
    print("Wrong ", a, b, a/b*100, '%')
    print(len(svm.support_vectors_))

    # 18,53

