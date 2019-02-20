'''
Script to optimize fire-arrival time estimation 
in 1 dimension using L^1 mimimization
Developed on Windows 10 using Python 2.7.x
by Lauren Hearn, February 2018
---
Dependencies:
numpy, matplotlib, pyomo
'''
import matplotlib.pyplot as plt
import numpy as np
import random as rd

n=10
h=1./n
eps = 0.1

def Bds(n): #artificial data generator
    Up = np.zeros(n)
    Lo = np.zeros(n)
    rand = np.zeros(n)
    x = np.zeros(n)
    s = np.zeros(n)
    for i in range(n):
        x[i] = i*h
        s[i] = abs(.5-x[i])
        rand[i] = rd.random()
        if rand[i] > 0.5:
            Up[i] = s[i] + rand[i]*eps
            Lo[i] = s[i] - rand[i]*eps
        else:
            Up[i] = 1
            Lo[i] = -1
    return Up, Lo

#test stuff
Up,Lo = Bds(n)
print(Up)
print(Lo)
print(h**2)

# Use Pyomo
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

m = ConcreteModel()

# Sets, Parameters, and Variables
m.N = RangeSet(0,n-1)
m.CN = RangeSet(1,n-2)

# Variables
m.u = Var(m.N, within=Reals)
m.a = Var(m.N, within=NonNegativeReals)
m.b = Var(m.N, within=NonNegativeReals)

# Constraints
# Upper and lower bounds 
#NOTE: think of a_i, b_i as slack variables
def LBound(m, i):
    return (-m.u[i-1] + 2*m.u[i] - m.u[i+1])/h**2 + m.a[i] >= 0
m.Lower = Constraint(m.CN, rule=LBound)

def UBound(m, i):
    return (-m.u[i-1] + 2*m.u[i] - m.u[i+1])/h**2 - m.b[i] <= 0
m.Upper = Constraint(m.CN, rule=LBound)

def X_Bounds(m, i):
    return (Lo[i], m.u[i], Up[i])
m.Xbound = Constraint(m.N, rule=X_Bounds)

# # Objective function
def ObjRule(m):
    return sum(m.a[i] for i in m.N) + sum(m.b[i] for i in m.N)
    
m.Obj = Objective(rule=ObjRule, sense=minimize)
opt = SolverFactory('glpk')
results = opt.solve(m)

print(results)

#plot results
for i in range(1,n):
    plt.scatter(i, m.u[i].value, c='black',marker='.')

plt.xlabel('Location x_i')
plt.ylabel('Fire Arrival time u(x_i)')
plt.title('1D L^1 Minimization')
plt.show()