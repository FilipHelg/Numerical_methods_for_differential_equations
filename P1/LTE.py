from IVP import IVP
import numpy as np
import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt

class LTE(IVP):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t, y):
        return self.A @ y
    
    def exact(self, t):
        return spl.expm(self.A * (t - self.t0)) @ self.y0