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
    def __init__(self, ivp, N, tol=1e-6, adaptive = False): #tol satt för fixed grid cases
        self.ivp = ivp
        self.u = ivp.y0.copy() #för att få med startvärdet
        self.N = N
        self.tol = tol
        self.adaptive = adaptive
        self.h = ((ivp.T - ivp.t0)/N)
        self.grid = np.linspace(ivp.t0, ivp.T, N + 1)
    
    def newstep(self):
        self.h = (self.tol / self.errest)**(2/(3*self.k)) * (self.tol / self.errestold)**(-1/(3*self.k))*self.h # given från (4)
        self.errestold = self.errest
    
    def integrate(self):
        if not self.adaptive:
            self.us = np.zeros((self.N+1, *self.ivp.y0.shape)) # för att få rätt form på arrayen
            self.us[0] = self.u
            
            for x in range(self.N):
                self.u = self.step(self.grid[x], self.u)
                self.us[x+1] = self.u
        else:
            self.h = (abs(self.ivp.T - self.ivp.t0) * (self.tol ** (1 / self.k)) / (100 * (1 + np.linalg.norm(self.ivp.f(self.ivp.t0, self.ivp.y0)))))
            self.grid = [self.ivp.t0] # måste göra listor som kan förlängas eftersom att vi inte vet antal steg när den är adaptiv
            self.us = [self.u.copy()] # initialvärdet (är samma som y0 eftersom den kopierades från y0 i __init__)
                                                                  
            while self.grid[-1] < self.ivp.T: 
                if self.grid[-1] + self.h >= self.ivp.T:#säkerställer att vi inte räknar för ett värde större än T, anpassar sista steget.
                    self.h = self.ivp.T - self.grid[-1] 
                    new_t = self.ivp.T 
                else:
                    new_t = self.grid[-1] + self.h
                
                self.u = self.step(self.grid[-1], self.u) # tar steg
                self.grid.append(new_t)
                self.us.append(self.u.copy())
                
                if new_t < self.ivp.T: # beräknar det nya steget som ska tas
                    self.newstep()

            self.grid = np.array(self.grid)
            self.us = np.array(self.us) # konvertera till vektorer för hantering senare
        
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
    
    
        
        
    
    
    