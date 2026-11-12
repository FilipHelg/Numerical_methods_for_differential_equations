from Solver import Solver
from LTE import LTE
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class RK4(Solver):
    def __init__(self, ivp, N):
        super().__init__(ivp, N)

    def step(self, tn, un):
        y1 = self.ivp.f(tn, un)
        y2 = self.ivp.f(tn + self.h/2, un + self.h * y1/2)
        y3 = self.ivp.f(tn + self.h/2, un + self.h * y2/2)
        y4 = self.ivp.f(tn + self.h, un + self.h * y3)
        return un + (self.h/6) * (y1 + 2*y2 + 2*y3 + y4)

def main():
    # Test case on sine functions
    N = 1000
    y0 = np.array([1,1])
    t0 = 0
    T = 10

    problem = LTE(t0, T, y0, np.array([[0, 1], [-1, 0]]))
    solver = RK4(problem, N)
    ts = np.linspace(t0, T, N + 1)
    us = np.zeros((N + 1, 2))
    errs = np.zeros((N + 1, 2))
    un = y0
    for (i,tn) in enumerate(ts):
        us[i] = un
        errs[i] = np.abs(problem.exact(tn) - un)
        un = solver.step(tn, un)
        

    us[N] = un
    plt.plot(ts, us)
    plt.plot(ts, errs)
    plt.show()
    print(sum(errs))



if __name__ == "__main__":
    main()