import numpy as np


class Evaluate:
    @staticmethod
    def get_error(y, p):
        s = 0
        for i in range(0, len(y)):
            s += (y[i] - p[i]) ** 2
        return np.square(s) / np.linalg.norm(p)
