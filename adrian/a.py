# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:00:27 2025

@author: xbfl0349
"""
import numpy as np
import matplotlib.pyplot as plt

Q = 0.99
alpha = 0.4709
beta = 1.0
H = 1.0
b_B = 1.178164343
N_points = 1001

def steady_state(Q, alpha, beta, H, b_B, N_points):
    delt_z = H / (N_points - 1)
    
    y = [b_B]
    
    for i in range(N_points - 1):
        new = delt_z * (alpha - Q / ( y[i]**3)) / beta + y[i]
        y.append(new)
    return np.array(y)

y = steady_state(Q, alpha, beta, H, b_B, N_points)
z_domain = np.linspace(0, H, N_points)
plt.xlabel("X")
plt.ylabel("Z")
plt.plot(np.array(y)/2, z_domain, color='black', label="Steady state solution")
plt.plot(-np.array(y)/2, z_domain, color='black')
plt.ylim((0.0, 1.1))
plt.legend()
plt.show()