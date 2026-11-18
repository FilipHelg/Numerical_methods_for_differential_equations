import numpy as np
import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt
from IVP import IVP
from LTE import LTE
from Solver import Solver
from RK34 import RK34
from Evaluator import Evaluator
from scipy.integrate import solve_ivp
import time


class VanDerPol(IVP):
    def __init__(self, t0, T, y0, mu):
        super().__init__(t0, T, y0)
        self.mu = mu
        
    def f(self, t, y):
        return np.array([y[1], self.mu * (1- y[0]**2) * y[1] - y[0]]) # van der pol ekvationen som given på sida 8



y0 = np.array([2 , 0])
mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 650, 1000] # för uppgift 4.3, 4.4 & 4.5, RK34 slutar fungera ungefär vid mu > 600
t0 = 0
x = 4.6 # vilken uppgift som ska lösas, 4.6 ger jämförelse mellan RK34, RK45, och ScipyBDF


if x == 4.1:
    mu = 100
    y0 = np.array([2 , 0])
    vanderpolproblem = VanDerPol(0, 2*mu, y0, mu) # test
    
    solver = RK34(vanderpolproblem, N = 100, tol = 1e-6, adaptive= True)
    solver.integrate()
    
    plt.figure()
    plt.title("y2 över tid, mu=100, y0 = [2,0]")
    plt.xlabel("tid")
    plt.ylabel("y2")
    
    plt.plot(solver.grid, solver.us[:, 1]) #ändra mellan 1&0 för att se y2 eller y1
    plt.show() 
    
    n = 5 #antalet 
    plt.figure() # 4.1
    plt.xlabel("y1")
    plt.ylabel("y2")
    for x in range(n): # n som alla hamnar på samma kurva.
        mu = 100
        y0 = np.array([(x+1)/5 , x*5])
        vanderpolproblem = VanDerPol(0, 2*mu, y0, mu)
            
        solver = RK34(vanderpolproblem, N = 100, tol = 1e-6, adaptive= True)
        solver.integrate()
        plt.plot(solver.us[:,0], solver.us[:,1], label = f"y0 = {y0}")
        plt.plot(y0[0], y0[1], 'ko')
    y0s = np.linspace(1,5, 5)/5
    plt.title(f"y2 mot y1, mu = {mu}, y0 = {y0s}") # alla hamnar på samma slutna kurva
    plt.show()

if x == 4.2:
    mu = 100
    y0 = np.array([2 , 0])
    vanderpolproblem = VanDerPol(0, 2*mu, y0, mu)
    solver = RK34(vanderpolproblem, N = 100, tol = 1e-6, adaptive= True)
    solver.integrate()
    plt.figure()
    plt.xlabel("time")
    plt.ylabel("stepsize")
    plt.semilogy(solver.grid [: -1], np.diff(solver.grid)) #4.2
    plt.title("Stepsizes over time")
    plt.show()
    
if x == 4.3:
    steps_rk34 = []

    for mu in mus:
        T_stiff = 0.7 * mu
        vdp_stiff = VanDerPol(t0, T_stiff, y0, mu)
    
        solver_rk34 = RK34(vdp_stiff, N=1, tol=1e-8, adaptive=True)
        solver_rk34.integrate()
        steps_rk34.append(len(solver_rk34.grid) - 1)

    plt.loglog(mus, steps_rk34, label="RK34", marker='o')
    plt.title("Total steps vs Stiffness parameter (mu)")
    plt.xlabel("Stiffness parameter (mu)")
    plt.ylabel("Number of steps (N)")
    plt.legend()
    plt.show()

if x == 4.4:
    mus = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000] # change size of mus to check whether larger mus take more time (it doesnt)
    musx10 = [100, 150, 220, 330, 470, 680, 1000, 1500, 2200, 3300, 4700, 6800, 10000]
    steps_bdf = []
    steps_bdfx10 = []
    start_time = time.perf_counter()
    
    for mu in mus:
        T_stiff = 0.7 * mu
        vdp_stiff = VanDerPol(t0, T_stiff, y0, mu)
            
        sol_bdf = solve_ivp(vdp_stiff.f, [0, T_stiff], y0, method='BDF', atol=1e-6, rtol=1e-6)
        steps_bdf.append(len(sol_bdf.t) - 1)
    end_time = time.perf_counter()
    result = (end_time - start_time)
    print(f"Total time was {result} seconds for the list: {mus}")
    
    start_time = time.perf_counter()
    for mu in musx10:
            T_stiff = 0.7 * mu
            vdp_stiff = VanDerPol(t0, T_stiff, y0, mu)
                
            sol_bdfx10 = solve_ivp(vdp_stiff.f, [0, T_stiff], y0, method='BDF', atol=1e-6, rtol=1e-6)
            steps_bdfx10.append(len(sol_bdf.t) - 1)
    end_time = time.perf_counter()
    result = (end_time - start_time)
    print(f"Total time was {result} seconds for the list: {musx10}")
    
    plt.loglog(mus, steps_bdf, label="SciPy BDF", marker='s', color='orange')
    plt.title("Total steps vs Stiffness parameter (mu)")
    plt.xlabel("Stiffness parameter (mu)")
    plt.ylabel("Number of steps (N)")
    plt.legend()
    plt.show()
       
if x == 4.5:
    steps_rk45 = []
    
    for mu in mus:
        T_stiff = 0.7 * mu
        vdp_stiff = VanDerPol(t0, T_stiff, y0, mu)
    
        sol_rk45 = solve_ivp(vdp_stiff.f, [0, T_stiff], y0, method='RK45', atol=1e-6, rtol=1e-6)
        steps_rk45.append(len(sol_rk45.t) - 1)

    plt.loglog(mus, steps_rk45, label="SciPy RK45", marker='^', color='green')
    plt.title("Total steps vs stiffness parameter (mu)")
    plt.xlabel("Stiffness parameter (mu)")
    plt.ylabel("Number of steps (N)")
    plt.legend()
    plt.show()

if x == 4.6:
    steps_rk34 = []
    steps_bdf = []
    steps_rk45 = []
    
    for mu in mus:
        T_stiff = 0.7 * mu
        vdp_stiff = VanDerPol(t0, T_stiff, y0, mu)
    
            
        solver_rk34 = RK34(vdp_stiff, N=1, tol=1e-8, adaptive=True)
        solver_rk34.integrate()
        steps_rk34.append(len(solver_rk34.grid) - 1)
        
        sol_bdf = solve_ivp(vdp_stiff.f, [0, T_stiff], y0, method='BDF', atol=1e-6, rtol=1e-6)
        steps_bdf.append(len(sol_bdf.t) - 1)
        
        sol_rk45 = solve_ivp(vdp_stiff.f, [0, T_stiff], y0, method='RK45', atol=1e-6, rtol=1e-6)
        steps_rk45.append(len(sol_rk45.t) - 1)

    # Plot total steps for each method in a collective graph
    plt.loglog(mus, steps_rk34, label="Custom RK34")
    plt.loglog(mus, steps_bdf, label="SciPy BDF")
    plt.loglog(mus, steps_rk45, label="SciPy RK45")
    
    plt.title("Metacomparison of total steps vs stiffness parameter")
    plt.xlabel("Stiffness parameter (mu)")
    plt.ylabel("Number of steps (N)")
    plt.legend()
    plt.show()
    
    
    
        