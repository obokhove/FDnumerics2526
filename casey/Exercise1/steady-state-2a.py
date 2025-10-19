# -*- coding: utf-8 -*-
#2a
"""
Steady-state dike profile from β b³ db/dz = α b³ − Q
Forward-Euler spatial integration and plot of (-b/2, b/2) vs z.
"""

import numpy as np
import matplotlib.pyplot as plt


# Parameters
alpha = 0.4709     # convection coefficient
beta = 1.0         # diffusion coefficient
Q = 0.99           # integration constant
H = 1.0            # total height (domain length)
bB = 1.178164343   # base width at z=0

# Numerical grid
N = 100              # number of grid points
z = np.linspace(0, H, N)
dz = z[1] - z[0]

# Allocate and initialize b array
b = np.zeros_like(z)
b[0] = bB  # boundary condition at base

# Forward Euler integration in space
# db/dz = (α/β) - (Q/(β b³))
for j in range(N - 1):
    db_dz = (alpha / beta) - (Q / (beta * b[j]**3))
    b[j + 1] = b[j] + dz * db_dz


# Plot the steady-state profile
plt.figure(figsize=(6, 8))

# dike shape: horizontal limits ±b/2
plt.plot(0.5 * b, z, ".", color='black', markersize=3)
plt.plot(-0.5 * b, z, ".", color='black', markersize=3)

plt.xlabel("Dike width (m)")
plt.ylabel("z (m)")
plt.title("Steady-state dike profile\n(Q=0.99, α=0.4709, β=1, H=1)")
plt.ylim(0, H)
plt.grid(True)
plt.tight_layout()
plt.show()
