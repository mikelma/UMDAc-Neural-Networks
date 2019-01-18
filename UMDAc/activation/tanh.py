import numpy as np

def tanh(a):
    return np.tanh(a)

if __name__ == '__main__':

    d = np.array([[1,3,-4]])
    print(d)
    print(tanh(d))
