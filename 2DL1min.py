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
#2d Case
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

#generate artificial data for 2d case
eps = 0.1
# p,n = size(u)
p = 10
n = 10
dx=1./p
dy=1./n

# def Bds(p,n):
#     Up = np.zeros((p,n))
#     Lo = np.zeros((p,n))
#     x = np.zeros((p,n))
#     s = np.zeros((p,n))
#     for i in range(p):
#         for j in range(n):
#             x[i,j] = [i*dx,j*dy]
#             s[i,j] = np.linalg.norm([.5,.5]-x[i,j])
#             if rd.random() > 0.5:
#                 Up[i,j] = s[i,j] + rd.random()*eps
#                 Lo[i,j] = s[i,j] - rd.random()*eps
#             else:
#                 Up[i,j] = 1
#                 Lo[i,j] = -1
#     return Up, Lo

# simple artificial data
def Bds(p,n):
    Up = np.zeros((p,n))
    Lo = np.zeros((p,n))
    for i in range(p):
        for j in range(n):
            if rd.random() > 0.5:
                Up[i,j] = 0.5 + rd.random()*eps
                Lo[i,j] = 0.5 - rd.random()*eps
            else:
                Up[i,j] = 1
                Lo[i,j] = -1
    return Up, Lo

#test stuff
Up,Lo = Bds(p,n)
print(Up)
print(Lo)

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
m.u = Var(m.M, m.N, within=Reals)
# Partials of u
m.u_xx = Var(m.M, m.N, within=Reals)
m.u_yy = Var(m.M, m.N, within=Reals)
m.u_xy = Var(m.M, m.N, within=Reals)
m.u_yx = Var(m.M, m.N, within=Reals)

m.a = Var(m.M, m.N, within=NonNegativeReals)
m.b = Var(m.M, m.N, within=NonNegativeReals)
m.eta = Var(m.M, m.N, within=NonNegativeReals)
m.xi = Var(m.M, m.N, within=NonNegativeReals)

# Constraints

# Build grad(u)
def Uxx(m,i,j):
    return m.u_xx[i,j] == ( (m.u[i+1,j]-m.u[i,j])/dx - (m.u[i,j]-m.u[i-1,j])/dx )/dx
m.Uxx = Constraint(m.CM, m.CN, rule=Uxx)

def Uyy(m,i,j):
    return m.u_yy[i,j] == ( (m.u[i,j+1]-m.u[i,j])/dy - (m.u[i,j]-m.u[i,j-1])/dy )/dy
m.Uyy = Constraint(m.CM, m.CN, rule=Uyy)

def Uxy(m,i,j):
    return m.u_xy[i,j] == ( (m.u[i+1,j+1]-m.u[i-1,j+1])/(2*dx) - (m.u[i+1,j-1]-m.u[i-1,j-1])/(2*dx) )/(2*dy)
m.Uxy = Constraint(m.CM, m.CN, rule=Uxy)

def Uyx(m,i,j):
    return m.u_yx[i,j] == ( (m.u[i+1,j+1]-m.u[i+1,j-1])/(2*dy) - (m.u[i-1,j+1]-m.u[i-1,j-1])/(2*dy) )/(2*dx)
m.Uyx = Constraint(m.CM, m.CN, rule=Uyx)

# Upper and lower bounds 
# NOTE: think of a_i, b_i as slack variables

def LBound(m,i,j): #todo - linearize these 2 constraints - maybe make into multiples?
    return abs(m.u_xx[i,j]) + abs(m.u_yy[i,j]) +abs(m.u_xy[i,j]) + abs(m.u_yx[i,j]) + m.a[i,j] >= 0
m.Lower = Constraint(m.CM, m.CN, rule=LBound)

def UBound(m,i,j):
    return abs(m.u_xx[i,j]) + abs(m.u_yy[i,j]) +abs(m.u_xy[i,j]) + abs(m.u_yx[i,j]) - m.b[i,j]<= 0
m.Upper = Constraint(m.CM, m.CN, rule=LBound)

def X_BoundU(m,i,j):
    return m.u[i,j] - Up[i,j] - m.xi[i,j] <= 0
m.XboundU = Constraint(m.M, m.N, rule=X_BoundU)

def X_BoundL(m,i,j):
    return m.u[i,j] - Lo[i,j] + m.eta[i,j] >= 0
m.XboundL = Constraint(m.M, m.N, rule=X_BoundL)

# # Objective function
def ObjRule(m):
    return (sum(sum(m.a[i,j] for i in m.M) for j in m.N) + sum(sum(m.b[i,j] for i in m.M) for j in m.N)
           + c1*sum(sum(m.xi[i,j] for i in m.M) for j in m.N) + c2*sum(sum(m.eta[i,j] for i in m.M) for j in m.N))
    
m.Obj = Objective(rule=ObjRule, sense=minimize)
opt = SolverFactory('glpk')
results = opt.solve(m)

print(results)

#plot results
# for i in range(1,n):
#     plt.scatter(i, m.u[i].value, c='black',marker='.')

# plt.xlabel('Location x_i')
# plt.ylabel('Fire Arrival time u(x_i)')
# plt.title('1D L^1 Minimization')
# plt.show()