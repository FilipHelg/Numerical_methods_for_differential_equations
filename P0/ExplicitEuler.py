import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from IVP import IVP
from LTE import LTE
from Solver import Solver

class ExplicitEuler(Solver):
    def __init__(self, ivp, u, N):
        Solver.__init__(self, ivp, u, N)
        return

    def step(self, tn, un):
        return un + self.h * self.ivp.f(tn, un)

def main():
    A = np.array ([[-1 , 1], [1 ,-3]])
    y0 = np.array([1, 2])
    ivp = LTE(t0 = 0, T = 1, y0 = y0, A = A)
    solver = ExplicitEuler(ivp, y0, N = 100)
    solver.integrate()
    # Plot the first component of the approximations { u_n }
    # vs. the temporal grid points { t_n }:
    plt.plot(solver.grid, solver.us[0, :])
    plt.show()

if __name__ == "__main__":
    main()