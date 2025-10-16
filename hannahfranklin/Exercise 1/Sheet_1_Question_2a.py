# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:18:07 2025

@author: qvbp0176
"""

## Numerics Exercise Sheet 1 - Question 2a Code

import numpy as np #import numpy for calculations
import matplotlib.pyplot as plt #import matplotlib for plotting results

#first define the constants as given:
Q = 0.99
a = 0.4709 #alpha value
b = 1.0 #beta value
H = 1 
b_B = 1.178164343 #b_0 value

#defining the grid space:
J=3001 #number of spatial steps
dz = H/J
z = np.linspace(0,H,J)
B = np.zeros((J,1))

B[0] = b_B #defining the initial value of b

for j in range (1,J):
    B[j] = B[j-1] + (a*dz)/b - (Q*dz)/(b*((B[j-1])**3))

fig1 = plt.figure()
plt.plot(B/2,z,'red')
plt.plot(-B/2,z,'red')
plt.xlabel('dike width')
plt.ylabel('z')
plt.title('Steady State Solution')

#print(B[J-1])