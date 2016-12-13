import pickle
import sys
import numpy as np

if __name__ == "__main__":

    data = None

    with open(sys.argv[1]) as f:
        global data
        data = pickle.load(f)
        print(data)
        data = pickle.load(f)
        print(data)
        data = pickle.load(f)
        print(data)
        data = pickle.load(f)
        print(data)



