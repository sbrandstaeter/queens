from __future__ import division

import numpy as np


# Non-monotonic Sobol G Function (8 parameters)
# First-order indices:
# x1: 0.7165
# x2: 0.1791
# x3: 0.0237
# x4: 0.0072
# x5-x8: 0.0001
def evaluate(values, a=None):

    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")
    if a is None:
        #a = [78,12,0.5,2,97,33]
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]

    ltz = np.array(values) < 0
    gto = np.array(values) > 1

    if ltz.any() == True:
        raise ValueError("Sobol G function called with values less than one")
    elif gto.any() == True:
        raise ValueError("Sobol G function called with values greater than one")

    output = 1

    for j in range(len(a)):
        x = values[j]
        output *= (abs(4 * x - 2) + a[j]) / (1 + a[j])

    return output
