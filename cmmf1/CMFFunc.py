import numpy as np
import matplotlib.pyplot as plt


class CMMFunc:
    def __init__(self, function):
        self.function = function

    def do_eval(self, x):
        res = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                res[i, j] = eval(self.function.replace('var', str(x[i, j])))
        return res

    def get_net_function(self, l, h):
        x = np.arange(0, l + h, h)
        N = len(x)
        x = x.reshape(N, 1)
        return x, self.do_eval(x)

    def __str__(self):
        return "f(x) = " + self.function.replace('var', 'x')
