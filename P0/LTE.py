import IVP
import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

class LTE(IVP.IVP):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t, y):
        return self.A @ y

    def exact(self, t):
        return scipy.linalg.expm(np.multiply.outer(np.asarray(t) - self.t0, self.A)) @ self.y0