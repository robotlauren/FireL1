import scipy.io as sio
import sys
import os
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random as rd

from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

if len(sys.argv) != 2:
    print 'Error: python %s matlab_file' % sys.argv[0]
    sys.exit(1)
else:
    matlab_file=sys.argv[1]
if (not os.path.isfile(matlab_file)) or (not os.access(matlab_file,os.R_OK)):
    print 'Error: file %s not exist or not readable' % sys.argv[1]
    sys.exit(1)

mat = loadmat(sys.argv[1])
case = sys.argv[1].split('/')[-1].split('.mat')[0]
X = mat['X']
Y = mat['Y']

X = np.delete(X, slice(None, None, 3)) # remove every 3rd data point
Y = np.delete(Y, slice(None, None, 3))

# Upper and Lower bounds for synthetic test
Up = np.array(mat['U']).astype(float)
Lo = np.array(mat['L']).astype(float)

Up = np.delete(Up, slice(None, None, 3))
Lo = np.delete(Lo, slice(None, None, 3))

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
fig1.suptitle("Plotting the data scaled to fit")
ax1.set_xlabel('Latitude')
ax1.set_ylabel('Longitude')
ax1.set_zlabel('Fire Arrival Time (days)')
ax1.scatter(X, Y, Up, c='red')
ax1.scatter(X, Y, Lo, c='blue')

p,n = Up.shape
print(p,n)
dx = 1./p
dy = 1./n

#epsilon as function of h
eps = 0.001*dx

#find constant values that work
c1 = 10
c2 = 50
c3 = 50

m = ConcreteModel()

# Sets, Parameters, and Variables

m.M = RangeSet(0,p-1) 
m.N = RangeSet(0,n-1)

m.VM = RangeSet(0,p-2) #v indeces
m.VN = RangeSet(0,n-2)

m.TM = RangeSet(0,p-3) #TGV indeces
m.TN = RangeSet(0,p-3)

# Variables
m.u = Var(m.M, m.N, within=NonNegativeReals)
m.v1 = Var(m.VM, m.VN, within=NonNegativeReals)
m.v2 = Var(m.VM, m.VN, within=NonNegativeReals) #new vector field v=(v1,v2)

# to take abs value out of obj fun
m.xnew1 = Var(m.M, m.N, within=Reals)
m.xnew2 = Var(m.M, m.N, within=Reals)
m.xnew3 = Var(m.M, m.N, within=Reals)
m.xnew4 = Var(m.M, m.N, within=Reals)
m.xnew5 = Var(m.M, m.N, within=Reals)

# Constraints
# soft constraints
m.eta = Var(m.M, m.N, within=NonNegativeReals)
m.xi = Var(m.M, m.N, within=NonNegativeReals)

# Upper and lower bounds
def X_BoundU(m,i,j):
    return m.u[i,j] - Up[i,j] - m.xi[i,j] <= 0
m.XboundU = Constraint(m.M, m.N, rule=X_BoundU)

def X_BoundL(m,i,j):
    return m.u[i,j] - Lo[i,j] + m.eta[i,j] >= 0
m.XboundL = Constraint(m.M, m.N, rule=X_BoundL)

# Absolute value constraints
def sum_1up(m,i,j):
    return (m.v1[i,j]-(m.u[i+1,j]-m.u[i,j])/dx) <= m.xnew1[i,j]
m.sum1up = Constraint(m.VM, m.VN, rule=sum_1up)

def sum_1down(m,i,j):
    return -(m.v1[i,j]-(m.u[i+1,j]-m.u[i,j])/dx) <= m.xnew1[i,j]
m.sum1down = Constraint(m.VM, m.VN, rule=sum_1down)

def sum_2up(m,i,j):
    return (m.v2[i,j]-(m.u[i,j+1]-m.u[i,j])/dy) <= m.xnew2[i,j]
m.sum2up = Constraint(m.VM, m.VN, rule=sum_2up)

def sum_2down(m,i,j):
    return -(m.v2[i,j]-(m.u[i,j+1]-m.u[i,j])/dy) <= m.xnew2[i,j]
m.sum2down = Constraint(m.VM, m.VN, rule=sum_2down)

def sum_3up(m,i,j):
    return ((m.v1[i+1,j]-m.v1[i,j])/dx) <= m.xnew3[i,j]
m.sum3up = Constraint(m.TM, m.TN, rule=sum_3up)

def sum_3down(m,i,j):
    return -((m.v1[i+1,j]-m.v1[i,j])/dx) <= m.xnew3[i,j]
m.sum3down = Constraint(m.TM, m.TN, rule=sum_3down)

def sum_4up(m,i,j):
    return (((m.v1[i,j+1]-m.v1[i,j])/dy+(m.v2[i+1,j]-m.v2[i,j])/dx)/2) <= m.xnew4[i,j]
m.sum4up = Constraint(m.TM, m.TN, rule=sum_4up)

def sum_4down(m,i,j):
    return -(((m.v1[i,j+1]-m.v1[i,j])/dy+(m.v2[i+1,j]-m.v2[i,j])/dx)/2) <= m.xnew4[i,j]
m.sum4down = Constraint(m.TM, m.TN, rule=sum_4down)

def sum_5up(m,i,j):
    return ((m.v2[i,j+1]-m.v2[i,j])/dy) <= m.xnew5[i,j]
m.sum5up = Constraint(m.TM, m.TN, rule=sum_5up)

def sum_5down(m,i,j):
    return -((m.v2[i,j+1]-m.v2[i,j])/dy) <= m.xnew5[i,j]
m.sum5down = Constraint(m.TM, m.TN, rule=sum_5down)

# new obj. func for TGV:
# made linear
def ObjRule(m):
    sum1 = sum(sum((m.xnew1[i,j]+m.xnew2[i,j])*dx*dy for i in m.VM) for j in m.VN)
    sum2 = sum(sum((m.xnew3[i,j]+m.xnew4[i,j]+m.xnew5[i,j])*dx*dy for i in m.TM) for j in m.TN)
    return c1*sum1 + sum2 + c2*sum(
        sum(m.xi[i,j] for i in m.M) for j in m.N)+c3*sum(
        sum(m.eta[i,j] for i in m.M) for j in m.N)
    
m.Obj = Objective(rule=ObjRule, sense=minimize)
opt = SolverFactory('glpk')
opt.options['dual'] # use Dual Simplex method
results = opt.solve(m)

print(results)

#plot results
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
z = np.zeros((p,n))
# v1 = np.zeros((p-1,n-1))
# v2 = np.zeros((p-1,n-1))
for i in range(p):
    for j in range(n):
        z[i,j] = m.u[i,j].value
ax2.plot_surface(X, Y, z, cmap='jet')
ax2.set_xlabel('Location x_ij')
ax2.set_zlabel('Fire Arrival time u(x_ij)')
ax2.set_title('2D L^1 Minimization')
plt.show()

# for i in range(p-1):
#     for j in range(n-1):
#         v1[i,j] = m.v1[i,j].value
#         v2[i,j] = m.v2[i,j].value

# #save output to matlab file
# data = {'u':z,'v1':v1,'v2':v2,'UpperBds':Up,'LowerBds':Lo,'dx':dx,'dy':dy}
# matback = savemat('syndata', data) # save u, v1, v2, dx, dy, Up, Lo 