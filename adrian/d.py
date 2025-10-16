# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:24:55 2025

@author: xbfl0349
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

Q = 0.99
alpha = 0.4709
beta = 1.0
H = 3.0
b_B = 1.178164343
zr0 = 0.9

"""---- Temporal Solution Parameters ----"""
N_points = 101
t_step = 0.00005
t_terminal = 3

import random

def random_color():
    r = random.randint(0, 255) /255
    g = random.randint(0, 255)/255
    b = random.randint(0, 255)/255
    return (r,g,b)

def exact_sol(Q, alpha, beta, H, zr0, N_points, t_step, terminal_time, t_plot_times):
    
    delt_z = H / (N_points - 1)
  
    """Saves time dependent boundary conditions in the form (b_B, b_t)"""
    b_bcs = []
    t = 0.0
    """Is of the form (z_values, b_values) for each time step"""
    result = []
    
    """Of the form (b_B, b_T)"""
    boundary_conditions = []
    while t <= terminal_time:
        
        """b between 0 and 1 inclusive to enforce physicality and validity for arctanh argument. 
        In addition f(b) is bijection and we will exploit this property to find corresponding values of z at fixed time t"""
        
        b_space = np.linspace(0, 1, N_points)
        z = []
        to_remove = []
        for i in range(len(b_space)):
            z_invert = beta / alpha * (b_space[i] - np.arctanh(b_space[i])) + zr0 + t
            if z_invert >= 0.0 and z_invert <= H:
                z.append(float(z_invert))
            else:
                to_remove.append(i)
        
        if t in t_plot_times:
            b_updated = np.delete(b_space, to_remove)
            result.append((z, b_updated))
        # if True in np.isnan(b):
        #     raise ValueError("ERROR: Solution diverges! Refine time step!")
        # '''--------- ROUNDING -------------'''
        t = round(t + t_step, 6)
    return result

def numerical_sol(Q, alpha, beta, H, b_B, b_T, N_points, t_step, terminal_time, exact_results):
    def p(index, b):
        return (b[index]**3 + b[index - 1]**3) / 2
    
    delt_z = H / (N_points - 1)
    b = np.array([b_T for i in range(N_points)])
    b[0] = b_B
    t = 0.0
    while t <= terminal_time:
        
        #print('start')
        for i in range(1, N_points - 1):
            convection = - 3 * alpha * b[i]**2 * (b[i]-b[i-1]) * t_step / delt_z
            diffusion = beta * t_step / delt_z**2 * (p(i + 1, b) * (b[i+1] - b[i]) - p(i, b) * (b[i] - b[i-1]))
            b[i] = convection + diffusion + b[i]
            #print(str(b[i]))
        if True in np.isnan(b):
            raise ValueError("ERROR: Solution diverges! Refine time step!")
        '''--------- ROUNDING -------------'''
        t = round(t + t_step, 6)
    return b

plot_times = [0.05, 0.1, 0.2, 0.5, 1., 2.]
results = exact_sol(Q, alpha, beta, H, zr0, N_points, t_step, t_terminal, plot_times)

"""Plotting Exact solution"""
for i in range(len(plot_times)):
    t = plot_times[i]
    b = results[i][1]
    z_domain = results[i][0]
    print(t)
    plt.title("Dike width with " + str(N_points) + " grid points and timestep " + str(t_step) + " seconds")
    col = random_color()
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.plot(b/2, z_domain, color=col, label="Exact solution at " + str(t) + " seconds", lw=4 )
    plt.plot(-b/2, z_domain, color=col, lw=4 )
    plt.ylim((0.0, H + 0.1))
    plt.legend()
    plt.show()