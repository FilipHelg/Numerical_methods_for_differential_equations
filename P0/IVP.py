import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

class IVP:

    t0 = None
    T = None
    y0 = None

    def __init__(self, t0, T, y0):
        self.t0 = t0
        self.T = T
        self.y0 = y0

    def f(t, y):
        raise NotImplementedError

    