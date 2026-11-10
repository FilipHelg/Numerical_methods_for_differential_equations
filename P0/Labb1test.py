import numpy as np
import scipy as sp
import scipy.linalg as spl
import matplotlib . pyplot as plt


#importerat från labb 0
class IVP():
    def __init__(self, t0, T, y0):
        self.t0 = t0
        self.T = T
        self.y0 = y0
    
    def f(t, y):
        raise NotImplementedError


class LTE(IVP):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t, y):
        return self.A @ y
    
    def exact(self, t):
        return spl.expm(self.A * (t - self.t0)) @ self.y0

class Solver():
    def __init__(self, ivp, N):
        self.ivp = ivp
        self.u = ivp.y0.copy()
        self.N = N
        self.h = ((ivp.T - ivp.t0)/N)
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
        
    def step(tn, un):
            raise NotImplementedError
        
class Evaluator():
    def __init__(self, ivp, Solver_class):
        self.ivp = ivp
        self.Solver_class = Solver_class
    
    def errvstime(self, N, relative = False):
        solver = self.Solver_class(self.ivp, N)
        solver.integrate()
        
        grid = solver.grid
        errs = np.zeros(N+1)
        
        for i in range(N+1):
            t_k = grid[i]
            u_k = solver.us[i]
            y_exact = self.ivp.exact(t_k)
            
            abs_error = np.linalg.norm(u_k-y_exact)

            if relative:
                errs[i] = abs_error / (np.linalg.norm(y_exact) + 1e-15)
            else:
                errs[i] = abs_error
        return grid, errs
    
    def errvsh(self, Ns, relative = False):
        errs = np.zeros(len(Ns))
        hs = np.zeros(len(Ns))
        y_exact_final = self.ivp.exact(self.ivp.T)
        norm_y_exact = np.linalg.norm(y_exact_final)
        
        for i in range(len(Ns)):
            solver = self.Solver_class(self.ivp, Ns[i])
            solver.integrate()
            
            hs[i] = solver.h
            u_final = solver.us[-1]
            abs_error = np.linalg.norm(u_final - y_exact_final)
            
            if relative:
                errs[i] = abs_error / (norm_y_exact+ 1e-15) #ifall den skulle vara 0
            else:
                errs[i] = abs_error
        return hs, errs    
    

############################################## BÖRJA IMPLEMENTERA RUNGEKUTTA letsago mario ##############################################################


class rungekutta(Solver): # gör för alla fyra ekvationer och lägg in viktning.
     def __init__(self, ivp, N):
        super().__init__(ivp, N)
        
    
    
    