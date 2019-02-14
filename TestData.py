#Create test data
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
            Up[i] = s[i] - rand[i]*eps
            Lo[i] = s[i] - rand[i]*eps
        else:
            Up[i] = -1
            Lo[i] = -1
    return Up, Lo

#test stuff
Up,Lo = Bds(n)
print(Up)
print(Lo)
print(h**2)