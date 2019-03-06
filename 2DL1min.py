'''
Script to optimize fire-arrival time estimation 
in 2 dimensions using L^1 mimimization
Developed on Windows 10 using Python 2.7.x
by Lauren Hearn, February 2018
---
Dependencies:
numpy, matplotlib, pyomo
(see README file)
'''
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
import random as rd

from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

#generate artificial data for 2d case
# p,n = size(u)
p = 200
n = 200
dx=1./p
dy=1./n

eps = 0.001*dx

# simple artificial data
def Bds(p,n):
    Up = np.zeros((p,n))
    Lo = np.zeros((p,n))
    for i in range(p):
        for j in range(n):
            if rd.random() > 0.5:
                Up[i,j] = 0.6 + rd.random()*eps
                Lo[i,j] = 0.4 - rd.random()*eps
            else:
                Up[i,j] = 1
                Lo[i,j] = 0
    return Up, Lo

#test stuff
Up,Lo = Bds(p,n)
x, y = np.meshgrid(np.linspace(0,1,p),np.linspace(0,1,n))

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
fig1.suptitle("Plotting the data scaled to fit")
ax1.scatter(x, y, Up, c='red')
ax1.scatter(x, y, Lo, c='blue')

#find constant values that work
c1 = 1
c2 = 1

m = ConcreteModel()

# Sets, Parameters, and Variables

m.M = RangeSet(0,p-1)
m.CM = RangeSet(1,p-2) #constraint indeces
m.N = RangeSet(0,n-1)
m.CN = RangeSet(1,n-2) #constraint indeces

# Variables
m.u = Var(m.M, m.N, within=Reals, initialize=1.0)

# soft constraints
m.eta = Var(m.M, m.N, within=NonNegativeReals)
m.xi = Var(m.M, m.N, within=NonNegativeReals)

# Constraints

# Upper and lower bounds - will need nl solver for this - looking into ipopt

def X_BoundU(m,i,j):
    return m.u[i,j] - Up[i,j] - m.xi[i,j] <= 0
m.XboundU = Constraint(m.M, m.N, rule=X_BoundU)

def X_BoundL(m,i,j):
    return m.u[i,j] - Lo[i,j] + m.eta[i,j] >= 0
m.XboundL = Constraint(m.M, m.N, rule=X_BoundL)

# Objective function
# maybe we don't need sqrt for minimization?
def ObjRule(m):
    return dx*dy*(
        sum(
            sum(sqrt(eps + (
                (-m.u[i-1,j] + 2*m.u[i,j] - m.u[i+1,j])/dx**2)**2 + (
                (-m.u[i,j-1] + 2*m.u[i,j] -m.u[i,j+1])/dy**2)**2 + 2*(
                (m.u[i+1,j+1] - m.u[i-1,j+1] - m.u[i+1,j-1] + m.u[i-1,j-1])/(4*dx*dy))**2) for i in m.CM) 
            for j in m.CN)) + c1*sum(sum(m.xi[i,j] for i in m.M) for j in m.N) + c2*sum(sum(m.eta[i,j] for i in m.M) for j in m.N)
    
m.Obj = Objective(rule=ObjRule, sense=minimize)
opt = SolverFactory('ipopt')
results = opt.solve(m)

print(results)

#plot results
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
z = np.zeros((p,n))
for i in range(1,p):
    for j in range(1,n):
        z[i,j] = m.u[i,j].value
ax2.plot_contour(x, y, z, cmap='jet')
ax2.set_xlabel('Location x_ij')
ax2.set_zlabel('Fire Arrival time u(x_ij)')
ax2.set_title('2D L^1 Minimization')
plt.show()