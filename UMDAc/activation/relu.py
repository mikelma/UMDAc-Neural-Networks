## ReLu activation function
## Version without derivate option
import numpy as np

def relu(X): ## Leaky ReLu activation function
        return np.maximum(X, 0., X)

