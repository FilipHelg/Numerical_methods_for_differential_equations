import numpy as np
import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt
from IVP import IVP
from LTE import LTE
from Evaluator import Evaluator
from Solver import Solver

class RK34(Solver): # gör för alla fyra ekvationer och lägg in viktning.
    def __init__(self, ivp, N, tol=1e-6, adaptive = False): #tol satt för fixed grid cases
        super().__init__(ivp, N, tol, adaptive)
        self.errest = 0
        self.errestold = tol 
        self.k = 4
    
    def step(self, tn, un):
        Y1 = self.ivp.f(tn, un)
        Y2 = self.ivp.f(tn + self.h/2, un + self.h * Y1/2)
        Y3 = self.ivp.f(tn + self.h/2, un + self.h * Y2/2)
        Z3 = self.ivp.f(tn+ self.h, un - self.h * Y1 + 2*self.h*Y2) # alla ekvationer givna under (3)
        Y4 = self.ivp.f(tn + self.h, un + self.h * Y3)
        
        localerror = self.h/6 * (2*Y2 + Z3 - 2*Y3 - Y4) # local error för l_n+1 enligt (3)
        self.errest = np.linalg.norm(localerror)
        return un + self.h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4) #beräknar u_n+1 genom metoderna anvisade i labben
    


###### test #####

if __name__ == "__main__":
    A = np.array([[0.0, 1.0], 
                  [-1.0, 0.0]])
    y0 = np.array([0.0, 1.0])
    t0 = 0.0
    periods = 5
    T = 2 * np.pi * periods
    
    ivp = LTE(t0, T, y0, A)
    
    solver = RK34(ivp, N=100, tol=1e-5, adaptive=True)
    solver.integrate()
    
    t_fine = np.linspace(t0, T, 500)
    sin_exact = np.sin(t_fine)
    
    plt.plot(t_fine, sin_exact, label = "sin(t)")
    plt.plot(solver.grid, solver.us[:, 0], 'o' , label='RK34', markersize = 5)
    plt.title("RK34 vs Sin")
    plt.xlabel("Tid")
    plt.ylabel("Värde")
    plt.show()