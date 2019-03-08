import scipy.io as sio
import sys
import os
from scipy.io import loadmat
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
# scale = mat['time_scale_num'][0]
X = mat['X']
Y = mat['Y']

# Upper and Lower bounds for synthetic test
Up = np.array(mat['U']).astype(float)
Lo = np.array(mat['L']).astype(float)

p,n = Up.shape
dx = 1./p
dy = 1./n

#epsilon as function of h
eps = 0.001*dx

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
ax2.contour(x, y, z, cmap='jet')
ax2.set_xlabel('Location x_ij')
ax2.set_zlabel('Fire Arrival time u(x_ij)')
ax2.set_title('2D L^1 Minimization')
plt.show()