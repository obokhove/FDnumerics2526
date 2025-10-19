# -*- coding: utf-8 -*-

#2b
"""
Time dependent solve for the convection-diffusion problem
"""

import numpy as np 
import matplotlib.pyplot as plt

#----------------------------------------------
# Steady-state solution
#----------------------------------------------

# Parameters
alpha = 0.4709     # convection coefficient
beta = 1.0         # diffusion coefficient
Q = 0.99           # integration constant
H = 1.0            # total height (domain length)
bB = 1.178164343   # base width at z=0
bT = 0.585373798   # top value (initial condition reference)

# Numerical grid for steady state
N_ss = 200
z_ss = np.linspace(0, H, N_ss)
dz_ss = z_ss[1] - z_ss[0]

b_ss = np.zeros_like(z_ss)
b_ss[0] = bB  # boundary condition at base

# Forward Euler integration in space
for j in range(N_ss - 1):
    db_dz = (alpha / beta) - (Q / (beta * b_ss[j]**3))
    b_ss[j + 1] = b_ss[j] + dz_ss * db_dz


#----------------------------------------------
# Time-dependent solver (Forward Euler)
#----------------------------------------------

def time_iteration(bPrev, alpha, b0, beta, dz, dt):
    # Forward Euler update for interior node
    bNext = (bPrev[1]
             - dt * (3 * alpha * b0**2 * (bPrev[1] - bPrev[0]) / dz
                     - beta * (bPrev[2] - 2*bPrev[1] + bPrev[0]) / dz**2))
    return bNext

def solve_time_dependent(alpha, beta, H, b0, bT, J, dt, max_time=2.5):
    dz = H / J
    print(f"J={J}, dz={dz:.4f}")

    dt_stable = dz**2 / (3 * dz * alpha * b0**2 + 2 * beta * b0**3)
    print(f"  Stable dt≈{dt_stable:.3e}, Given dt={dt:.3e}")

    z_values = np.linspace(0, H, J + 1)

    # Initial condition: perturbation = 0
    b_values = [np.zeros(J + 1)]
    b_values[0][0] = b0 - bT  # boundary condition at base
    b_values[0][-1] = 0       # boundary condition at top (zero perturbation)

    time_steps = int(max_time / dt)
    for _ in range(time_steps):
        b_prev = b_values[-1]
        b_next = np.zeros_like(b_prev)
        b_next[0] = b0 - bT
        b_next[-1] = 0
        for j in range(1, J):
            b_next[j] = time_iteration(b_prev[j-1:j+2], alpha, b0, beta, dz, dt)
        b_values.append(b_next)

    time_values = np.arange(0, max_time + dt, dt)

    # add back bT baseline
    b_values = np.array(b_values) + bT

    return z_values, time_values, b_values


#----------------------------------------------
# Run simulations for multiple grid resolutions
#----------------------------------------------

def plot_results(J, dt, max_time=2.5):
    z, t_values, b_values = solve_time_dependent(alpha, beta, H, bB, bT, J, dt, max_time)

    # times to plot
    plot_times = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    colors = ['r', 'g', 'b', 'c', 'm', 'k']

    plt.figure(figsize=(6, 8))

    for i, t_target in enumerate(plot_times):
        n_idx = np.argmin(np.abs(t_values - t_target))
        b_plot = b_values[n_idx, :]
        plt.scatter(0.5*b_plot, z, color=colors[i], s=10, label=f"t={t_target}")
        plt.scatter(-0.5*b_plot, z, color=colors[i], s=10)

    # steady-state profile
    plt.plot(0.5*b_ss, z_ss, 'k--', linewidth=1.2, label='steady state')
    plt.plot(-0.5*b_ss, z_ss, 'k--', linewidth=1.2)

    plt.xlabel("Dike width (m)")
    plt.ylabel("z (m)")
    plt.title(f"Nonlinear convection–diffusion evolution\nForward Euler, J={J}, Δt={dt:.1e}")
    plt.ylim(0, H)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#----------------------------------------------
# Generate plots for J = 11, 21, 41
#----------------------------------------------
plot_results(11, 1e-4)
plot_results(21, 1e-4)
plot_results(41, 1e-4)
#plot_results(81, 1e-5)