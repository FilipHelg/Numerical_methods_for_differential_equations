import numpy as np
import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt
from IVP import IVP
from LTE import LTE
from Solver import Solver
from RK34 import RK34
from Evaluator import Evaluator


class vanderpol(IVP):
    def __init__(self, t0, T, y0, mu):
        super().__init__(t0, T, y0)
        self.mu = mu
        
    def f(self, t, y):
        return np.array([y[1], self.mu * (1- y[0]**2) * y[1] - y[0]]) # van der pol ekvationen som given på sida 8
    

if __name__ == "__main__":
    mu = 100
    y0 = np.array([2 , 0])
    vanderpolproblem = vanderpol(0, 2*mu, y0, mu)
    
    solver = RK34(vanderpolproblem, N = 100, tol = 1e-6, adaptive= True)
    solver.integrate()
    
    plt.figure()
    plt.title("y2 över tid, mu=100, y0 = [2,0]")
    plt.xlabel("tid")
    plt.ylabel("y2")
    plt.plot(solver.grid, solver.us[:, 1])
    plt.show() 
    
    plt.figure()
    plt.xlabel("y1")
    plt.ylabel("y2")
    for x in range(5):
        mu = 100
        y0 = np.array([(x+1)/5 , 0])
        vanderpolproblem = vanderpol(0, 2*mu, y0, mu)
            
        solver = RK34(vanderpolproblem, N = 100, tol = 1e-6, adaptive= True)
        solver.integrate()
        plt.plot(solver.us[:,0], solver.us[:,1], label = f"y0 = {y0}")
        plt.plot(y0[0], y0[1], 'ko')
    y0s = np.linspace(1,5, 5)/5
    plt.title(f"y2 mot y1, mu = {mu}, y0 = {y0s}") # alla hamnar på samma slutna kurva
    plt.show()

    
    
    
    
        