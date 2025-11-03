# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:28:06 2025

@author: py21kt
"""

import numpy as np
import matplotlib.pyplot as plt

Q = 0.99 # integration constant
alpha = 0.4709 # convection coefficient
beta = 1.0 # diffusion coefficient
H = 1 # Height of the dike
b_B = 1.178164343 # boundary condition at the base
b_T = 0.585373798
grid_points = 41

dz = H / (grid_points-1)
time_step = 0.1 * (dz)**2
time_max = 1.0

z = np.linspace(0,H,grid_points)
b = np.zeros(grid_points)  #stores numerical solution for b(z)

b[0] = b_B
b[-1] = b_T #initialise with boundary condition then update using numerical technique

# Euler forward Integration 
def steady_state(grid_points):
    z = np.linspace(0, H, grid_points)
    b = np.zeros(grid_points)
    b[0] = b_B
    b[-1] = b_T
    for i in range(grid_points-1):
        derivative = (alpha * (b[i])**3 - Q) / (beta * (b[i]**3))
        b[i+1] = b[i] + (H/(grid_points-1)) * derivative
        
    return z, b
    

 
    
def nonlinear_PDE (grid_points, dt, t_max):
    z = np.linspace(0, H, grid_points)
    b = np.full(grid_points, b_T)
    b[0]=b_B
    t = 0
    while t <t_max:
        b_new = b.copy()
        for i in range (1, grid_points - 1):
            convective_term = alpha * ((b[i+1]**3 - b[i-1]**3) / (2* dz))
            #print("Convective term:", convective_term)
            
            
            # Compute midpoint values of b between j and j+1, and j-1 and j
            b_jph = 0.5 * (b[i] + b[i + 1])  # midpoint between j and j+1
            b_jmh = 0.5 * (b[i - 1] + b[i])  # midpoint between j-1 and j
            
            # Compute gradients at midpoints (finite differences)
            db_jph = (b[i + 1] - b[i]) / dz  # forward difference at j+1/2
            db_jmh = (b[i] - b[i - 1]) / dz  # backward difference at j-1/2
            
            # Compute nonlinear fluxes at midpoints
            # These represent the flux of b^3 * db/dz at the cell interfaces
            flux_ph = beta * b_jph**3 * db_jph  # flux at j+1/2
            flux_mh = beta * b_jmh**3 * db_jmh  # flux at j-1/2
            
            # Compute the divergence of the flux (i.e., net diffusive effect at j)
            diffusive_term = (flux_ph - flux_mh) / dz
            
                        
                           
            
            #print("Diffusive term:", diffusive_term)

            
            
            
            b_new[i] = b[i] - dt* (convective_term - diffusive_term)
            #b_new = np.nan_to_num(b_new, nan=b_T, posinf=b_T, neginf=b_T)
        
        
       
        b = b_new
        b_new[0] = b_B
        b_new[-1] = b_T
        t = t+dt
    return z, b 

#Running the solver and computing the steady-state

dt = 0.01* (dz**2)



z_005, b_005 = nonlinear_PDE(grid_points, dt, 0.05)
z_01, b_01 = nonlinear_PDE(grid_points, dt, 0.1)
z_02, b_02 = nonlinear_PDE(grid_points, dt, 0.2)
z_05, b_05 = nonlinear_PDE(grid_points, dt, 0.5)
z_1, b_1 = nonlinear_PDE(grid_points, dt, 1)
z_2, b_2 = nonlinear_PDE(grid_points, dt, 2)


z_steady, b_steady = steady_state(3000)

# Compute dike width boundaries
x_left = -b / 2
x_right = b / 2


x_steady_left = -b_steady / 2
x_steady_right = b_steady / 2





plt.figure(figsize=(6, 8))

# Plot transient solutions — only label one curve per time
plt.plot(-b_005/2, z_005, color='r')
plt.plot(b_005/2, z_005, color='r', label='t = 0.05')

plt.plot(-b_01/2, z_01, color='g')
plt.plot(b_01/2, z_01, color='g', label='t = 0.1')

plt.plot(-b_02/2, z_02, color='m')
plt.plot(b_02/2, z_02, color='m', label='t = 0.2')

plt.plot(-b_05/2, z_05, color='c')
plt.plot(b_05/2, z_05, color='c', label='t = 0.5')

plt.plot(-b_1/2, z_1, color='blue')
plt.plot(b_1/2, z_1, color='blue', label='t = 1.0')

plt.plot(-b_2/2, z_2, color='y')
plt.plot(b_2/2, z_2, color='y', label='t = 2.0')

# Plot steady state — only one label
plt.plot(x_steady_left, z_steady, color='black', linestyle='--')
plt.plot(x_steady_right, z_steady, color='black', linestyle='--', label='Steady state')

plt.xlabel('Dike width (m)')
plt.ylabel('Height z (m)')
plt.title('Dike Profile Evolution (N = 41)')
plt.legend(title='Legend', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()










