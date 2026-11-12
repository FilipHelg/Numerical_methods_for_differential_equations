from IVP import IVP
from RK34 import RK34
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class LotkaVolterra(IVP):
    def __init__(self, t0, T, y0, a, b, c, d):
        super().__init__(t0, T, y0)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f(self, t, y):
        return np.array([self.a * y[0] - self.b*y[0]*y[1], self.c*y[0]*y[1] - self.d*y[1]])

def main():
    #Test case as per 3.2
    a = 3
    b = 9
    c = 15
    d = 15
    y0 = np.array([1,1]) # Equilibrium at [d/c, b/a]. Solution and its period changes drastically with the initial condition.
    t0 = 0
    T = 120
    N = 1000
    tol = 1e-2 #choose 1e-6 for good result, or 1e-2 to see drift

    def H(y):
        return c*y[0] + b*y[1] - d*np.log(y[0]) - a*np.log(y[1])

    problem = LotkaVolterra(t0, T, y0, a, b, c, d)
    solver = RK34(problem, N, tol)

    ts = np.linspace(t0, T, N + 1)
    us = np.zeros((N + 1, 2))
    #errs = np.zeros((N + 1, 2))
    un = y0
    relErrHs = np.zeros(N+1)
    H0 = H(y0)
    for (i,tn) in enumerate(ts):
        us[i] = un
        #errs[i] = np.abs(problem.exact(tn) - un)
        relErrHs[i] = np.abs(H(un)/H0 - 1)
        un = solver.step(tn, un)
        

    us[N] = un
    plt.plot(ts, us)
    plt.plot(ts, relErrHs)
    plt.show()
    #print(sum(errs))

if __name__ == "__main__":
    main()