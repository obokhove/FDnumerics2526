# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:29:22 2025

@author: py21kt
"""

import numpy as np
import matplotlib.pyplot as plt

Q = 0.99 # integration constant
alpha = 0.4709 # convection coefficient
beta = 1.0 # diffusion coefficient
H = 1 # Height of the dike
b_B = 1.178164343 # boundary condition at the base
grid_points = 1000 

z = np.linspace(0,H,grid_points)
b = np.zeros(grid_points)  #stores numerical solution for b(z)

b[0] = b_B #initialise with boundary condition then update using numerical technique

# Euler forward Integration 
def steady_state(grid_points):
    z = np.linspace(0, H, grid_points)
    b = np.zeros(grid_points)
    b[0] = b_B
    for i in range (grid_points - 1):
        derivative = (alpha * (b[i])**3 - Q) / (beta * (b[i]**3))
        b[i+1] = b[i] + (H/(grid_points-1)) * derivative
        
    return z, b
        



# Compute dike width boundaries
x_left = -b / 2
x_right = b / 2

# Plotting the dike profile as a line graph
plt.figure(figsize=(6, 8))
plt.plot(x_left, z, label='Left boundary', color='blue')
plt.plot(x_right, z, label='Right boundary', color='red')
plt.xlabel('Dike width (m)')
plt.ylabel('Height z (m)')
plt.title('Steady-State Dike Profile')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("steady_state_dike_profile.png")
plt.show()

