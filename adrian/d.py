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
    
    """Of the form [b_B at t = 0, ...]"""
    boundary_conditions = []
    while t <= terminal_time:
        
        """b between 0 and 1 inclusive to enforce physicality and validity for arctanh argument. 
        In addition f(b) is bijection and we will exploit this property to find corresponding values of z at fixed time t"""
        
        b_space = np.linspace(0, 1, N_points)
        z = []
        to_remove = []
        for i in range(len(b_space)):
            z_invert = beta / alpha * (b_space[i] - np.arctanh(b_space[i])) + zr0 + alpha*t
            if z_invert >= 0.0 and z_invert <= H:
                z.append(float(z_invert))
            else:
                to_remove.append(i)
        
        b_updated = np.delete(b_space, to_remove)
        """Finding bottom boundary condittions: 
            All solutions b[i] for z > zr0 + c * t are 0 by maximum principle and do not have to be saved here"""
        b_B = float(b_updated[z.index(np.min(z))])
        boundary_conditions.append(b_B)
        if t in t_plot_times:
            result.append((np.array(z), b_updated))
        # if True in np.isnan(b):
        #     raise ValueError("ERROR: Solution diverges! Refine time step!")
        # '''--------- ROUNDING -------------'''
        t = round(t + t_step, 6)
    return boundary_conditions, result

def numerical_sol(Q, alpha, beta, zr0, H, N_points, t_step, terminal_time, boundary_conditions, plot_times, initial_conditions):
    def p(index, b):
        return (b[index]**3 + b[index - 1]**3) / 2
    
    delt_z = H / (N_points - 1)
    t = 0.0
    count = 0
    result = []
    b = np.array([0.0 for i in range(N_points)])
    
    """ Setting up initial conditions based on exact solution!"""
    for i in range(N_points):
        z = i * delt_z
        index = np.abs(initial_conditions[0] - z).argmin()
        b[i] = initial_conditions[1][index]
    print(b)
    while t <= terminal_time:
        
        b_B = boundary_conditions[count]
        b[0] = b_B
        #print('start')
        for i in range(1, N_points - 1):
            
            """ Enforcing Boundary condition:
            -> Everything above z* = t * alpha + zr0 
            results in negative values of b which are ignored by maximum principle
            """
            if i * delt_z >= t * alpha + zr0:
                b[i] = 0.0
                continue
            convection = - 3 * alpha * b[i]**2 * (b[i]-b[i-1]) * t_step / delt_z
            diffusion = beta * t_step / delt_z**2 * (p(i + 1, b) * (b[i+1] - b[i]) - p(i, b) * (b[i] - b[i-1]))
            b[i] = convection + diffusion + b[i]
            #print(str(b[i]))
        if True in np.isnan(b):
            raise ValueError("ERROR: Solution diverges! Refine time step!")
        if t in plot_times:
            print("Solving Numerical solution at " + str(t) + " seconds!")
            result.append(b.copy())
        '''--------- ROUNDING -------------'''
        t = round(t + t_step, 6)
        count += 1
    return result

"""------------------ EXECUTION ---------------------"""
"""Make sure plot times start at 0.0!!! To include initial conditions for numerical solution"""
plot_times = [0.0, 0.05, 0.1, 0.2, 0.5, 1., 2., 3.0]
boundary_conditions, results = exact_sol(Q, alpha, beta, H, zr0, N_points, t_step, t_terminal, plot_times)
num_solution = numerical_sol(Q, alpha, beta, zr0, H, N_points, t_step, t_terminal, boundary_conditions, plot_times, results[0])

"""Plotting Exact solution"""
for i in range(len(plot_times)):
    t = plot_times[i]
    b = results[i][1]
    z_domain = results[i][0]
    plt.title("Dike width with " + str(N_points) + " grid points and timestep " + str(t_step) + " seconds")
    col = random_color()
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.plot(b/2, z_domain, color=col, label="Exact solution at " + str(t) + " seconds", lw=4 )
    plt.plot(-b/2, z_domain, color=col, lw=4 )
    plt.ylim((0.0, H + 0.1))
    plt.legend()
    plt.show()
    
"""Plotting Exact solution VS Numerical Solution"""
for i in range(len(plot_times)):
    t = plot_times[i]
    b = results[i][1]
    b_num = num_solution[i]
    z_domain = results[i][0]
    z_domain_num = np.linspace(0, H, N_points)
    plt.title("Dike width with " + str(N_points) + " grid points and timestep " + str(t_step) + " seconds")
    col = random_color()
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.plot(b/2, z_domain, color='black', label="Exact solution at " + str(t) + " seconds", lw=4 )
    plt.plot(-b/2, z_domain, color='black', lw=4 )
    plt.plot(b_num/2, z_domain_num, color=col, label="NUMERICAL Solution at " + str(t) + " seconds", lw=4 )
    plt.plot(-b_num/2, z_domain_num, color=col, lw=4 )
    plt.ylim((0.0, H + 0.1))
    plt.legend()
    plt.show()