import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from IVP import IVP
from LTE import LTE

class Solver():
    def __init__(self, ivp, u, N):
        self.ivp = ivp
        self.u = u
        self.N = N
        self.grid = np.linspace(ivp.t0, ivp.T, N+1)
        self.h = (ivp.T - ivp.t0) / N
        self.us = np.array([])
        return

    def step(self, tn, un):
        raise NotImplementedError

    def integrate(self):
        self.us = np.zeros((len(self.u), self.N + 1))
        for i in range(self.N):
            self.us[:, i] = self.u
            self.u = self.step(self.grid[i], self.u)

        self.us[:, self.N] = self.u
        return self.grid, self.us
