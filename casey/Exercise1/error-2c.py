# -*- coding: utf-8 -*-
"""
Error approximation based on steady state solution
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters
alpha = 0.4709     # convection coefficient
beta = 1.0         # diffusion coefficient
Q = 0.99           # integration constant
H = 1.0            # total height (domain length)
bB = 1.178164343   # base width at z=0

# ------------------------------------------------------------
# Function to compute steady-state profile for given N
def compute_profile(N):
    z = np.linspace(0, H, N)
    dz = z[1] - z[0]
    b = np.zeros_like(z)
    b[0] = bB
    for j in range(N - 1):
        db_dz = (alpha / beta) - (Q / (beta * b[j]**3))
        b[j + 1] = b[j] + dz * db_dz
    return z, b

# ------------------------------------------------------------
# High-resolution reference solution ("exact")
N_ref = 1000
z_ref, b_ref = compute_profile(N_ref)

# ------------------------------------------------------------
# Test resolutions
N_values = [25, 50, 100, 200, 400]
L2_errors = []
Linf_errors = []

for N in N_values:
    z, b = compute_profile(N)

    # Interpolate high-resolution reference onto coarse grid
    b_exact = np.interp(z, z_ref, b_ref)

    # Error
    e = b_exact - b

    # Composite trapezoidal rule for L2 norm
    dz = z[1] - z[0]
    L2 = np.sqrt(np.trapezoid(e**2, z))     # equivalent to composite trapezoidal
    Linf = np.max(np.abs(e))

    L2_errors.append(L2)
    Linf_errors.append(Linf)

# ------------------------------------------------------------
# Display results
print(f"{'N':>6} | {'L2 error':>12} | {'L∞ error':>12}")
print("-" * 36)
for N, L2, Linf in zip(N_values, L2_errors, Linf_errors):
    print(f"{N:6d} | {L2:12.6e} | {Linf:12.6e}")

# ------------------------------------------------------------
# Estimate empirical convergence order (p)
orders_L2 = []
orders_Linf = []
for i in range(1, len(N_values)):
    r = N_values[i] / N_values[i - 1]
    p_L2 = np.log(L2_errors[i - 1] / L2_errors[i]) / np.log(r)
    p_Linf = np.log(Linf_errors[i - 1] / Linf_errors[i]) / np.log(r)
    orders_L2.append(p_L2)
    orders_Linf.append(p_Linf)

print("\nEstimated convergence order (L2): ", np.mean(orders_L2))
print("Estimated convergence order (L∞): ", np.mean(orders_Linf))

# ------------------------------------------------------------
# Optional: plot errors on log-log scale
plt.figure()
plt.loglog(N_values, L2_errors, 'o-', label=r"$L_2$")
plt.loglog(N_values, Linf_errors, 's-', label=r"$L_\infty$")
plt.xlabel("Number of grid points (N)")
plt.ylabel("Error norm")
plt.title("Convergence of steady-state integration")
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

"""
The solution shows an approximate first order accuracy in the spatial discretization.
"""
