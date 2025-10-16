# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 09:32:46 2025

@author: xbfl0349
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from a import steady_state

def random_color():
    r = random.randint(0, 255) /255
    g = random.randint(0, 255)/255
    b = random.randint(0, 255)/255
    return (r,g,b)

Q = 0.99
alpha = 0.4709
beta = 1.0
H = 1.0
b_B = 1.178164343
b_T =  0.585373798


plot_times = [0.05, 0.1, 0.2, 0.5, 1., 2.]
t_step = 0.00005
N_points = 41 #11, 21, 41
delt_z = H / (N_points - 1)

b = np.array([b_T for i in range(N_points)])
b[0] = b_B
b_steady = steady_state(Q, alpha, beta, H, b_B, 4001)
z_domain_steady = np.linspace(0, H, 4001)
z_domain = np.linspace(0, H, N_points)

t = 0
# Half time step at (index - 1/2) for p_{index - 1/2} = (p_index - p_{index - 1}) / 2
def p(index, b):
    return (b[index]**3 + b[index - 1]**3) / 2

while t <= plot_times[-1]:
    
    #print('start')
    for i in range(1, N_points - 1):
        convection = - 3 * alpha * b[i]**2 * (b[i]-b[i-1]) * t_step / delt_z
        diffusion = beta * t_step / delt_z**2 * (p(i + 1, b) * (b[i+1] - b[i]) - p(i, b) * (b[i] - b[i-1]))
        b[i] = convection + diffusion + b[i]
        #print(str(b[i]))
        
    if t in plot_times:
        print(t)
        plt.title("Dike width with " + str(N_points) + " grid points and timestep " + str(t_step) + " seconds")
        col = random_color()
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.plot(b/2, z_domain, color=col, label="Solution at " + str(t) + " seconds", lw=4 )
        plt.plot(-b/2, z_domain, color=col, lw=4 )
        plt.plot(b_steady/2, z_domain_steady, color='black', label="Steady state solution")
        plt.plot(-b_steady/2, z_domain_steady, color='black')
        plt.ylim((0.0, 1.1))
        plt.legend()
        plt.show()
        
        
    
    '''--------- ROUNDING -------------'''
    t = round(t + t_step, 6)