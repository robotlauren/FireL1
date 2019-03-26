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
scale = mat['time_scale_num'][0]
LON = mat['fxlon']
lon = np.reshape(LON,np.prod(LON.shape)).astype(float)
LAT = mat['fxlat']
lat = np.reshape(LAT,np.prod(LAT.shape)).astype(float)
T = np.array(mat['T']).astype(float)
U = np.array(mat['U']).astype(float)
L = np.array(mat['L']).astype(float)
#time scaling (to days)
tscale = 24*3600
U = U/tscale
L = L/tscale
# make sure all upper bounds > lower bounds
print 'U>L: ',(U>L).sum()
print 'U<L: ',(U<L).sum()
print 'U==L: ',(U==L).sum()

uu = np.reshape(U,np.prod(U.shape))
ll = np.reshape(L,np.prod(L.shape))
# create mask
tt = np.reshape(T,np.prod(T.shape))
mk = tt==scale[1]-scale[0]
# make values outside of mask large
uu[mk] = uu.max()
ll[mk] = uu.max()-.2

# final Upper and Lower bounds
Up = np.reshape(uu, U.shape)
Lo = np.reshape(ll, L.shape)
p,n = Up.shape
dx = 1./p
dy = 1./n

#epsilon as function of h
eps = 0.001*dx

#find constant values that work
c1 = 1
c2 = 1
c3 = 1
c4 = 1

m = ConcreteModel()

# Sets, Parameters, and Variables

m.M = RangeSet(0,p-1)
m.CM = RangeSet(1,p-2) #constraint indeces
m.N = RangeSet(0,n-1)
m.CN = RangeSet(1,n-2) #constraint indeces

# Variables
m.u = Var(m.M, m.N, within=Reals)
m.v1 = Var(m.M, m.N, within=Reals)
m.v2 = Var(m.M, m.N, within=Reals) #new vector field v=(v1,v2)

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

# new obj. func for TGV:
# note: eps under sqrt to avoid sqrt(0)
def ObjRule(m):
    sum1 = sum(
        sum(sqrt(eps +(m.v1[i,j]-(m.u[i+1,j]-m.u[i-1,j])/(2*dx))**2+(m.v2[i,j]-(m.u[i,j+1]-m.u[i,j-1])/(2*dy))**2)*dx*dy 
            for i in m.CM) for j in m.CN)
    sum2 = sum(
        sum(sqrt(eps +((m.v1[i+1,j] + m.v1[i-1,j])/(2*dx))**2+((m.v1[i,j+1]-m.v1[i,j-1])/(2*dy)+(m.v2[i+1,j]-m.v2[i-1,j])/(2*dx))**2/2+((m.v2[i,j+1]-m.v2[i,j-1])/(2*dy))**2)*dx*dy for i in m.CM) for j in m.CN)
    return c1*sum1 + sum2 + c2*sum(sum(m.xi[i,j] for i in m.M) for j in m.N)+c3*sum(sum(m.eta[i,j] for i in m.M) for j in m.N)
    
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
ax2.contour(LON, LAT, z, levels=100, cmap='jet')
ax2.set_xlabel('Location x_ij')
ax2.set_zlabel('Fire Arrival time u(x_ij)')
ax2.set_title('2D L^1 Minimization')
plt.show()