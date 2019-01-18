#Sigmoid activation function

import numpy as np

def sigmoid(x, deriv = False):
    if deriv == True:
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid*(1-sigmoid) 
    else:
        return 1 / (1 + np.exp(-x))
