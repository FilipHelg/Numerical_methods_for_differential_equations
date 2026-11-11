import numpy as np
import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt
from IVP import IVP
from LTE import LTE

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