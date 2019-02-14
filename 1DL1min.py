import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os

# Use Pyomo
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

m = ConcreteModel()

# Sets, Parameters, and Variables
m.N = RangeSet(0,n+1)

# Variables
m.u = Var(m.N, within=Reals)
m.a = Var(m.N, within=NonNegativeReals)
m.b = Var(m.N, within=NonNegativeReals)

# Constraints
# Upper and lower bounds 
#NOTE: think of a_i, b_i as slack variables
def LBound(m, i):
    return (-m.u[i-1] + 2*m.u[i] - m.u[i+1])/h**2 + m.a[i] >= 0
m.Size = Constraint(m.N, rule=LBound)

def UBound(m, i):
    return (-m.u[i-1] + 2*m.u[i] - m.u[i+1])/h**2 - m.b[i] <= 0
m.Size = Constraint(m.N, rule=LBound)

def X_Bounds(m, i):
    return (Lo[i], m.u[i], Up[i])
m.Xbound = Constraint(m.N, rule=X_Bounds)

# # Objective function
def ObjRule(m):
    return sum(m.a[i] for i in m.N) + sum(m.b[i] for i in m.N)

# def ObjRule(m):
#     return sum(abs(-m.u[i-1] + 2*m.u[i] - m.u[i+1]/h^2) for i in m.N)
    
m.Obj = Objective(rule=ObjRule, sense=minimize)
opt = SolverFactory('glpk')
results = opt.solve(m)

print results