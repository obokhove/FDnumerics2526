# L2 Error vs Grid Spacing (log-log plot)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Physical and numerical parameters
alpha = 0.4709
beta = 1.0
H = 1.0
b_B = 1.178164343
b_T = 0.585373798
Q = 0.99
t_max = 2.0
dt = 1e-4


def compute_steady_state(N):
    dz = H / (N - 1)
    b = np.zeros(N)
    b[0] = b_B
    for j in range(N - 1):
        db_dz = (alpha * b[j]**3 - Q) / (beta * b[j]**3)
        b[j + 1] = b[j] + dz * db_dz
    b[-1] = b_T
    return b


def solve_nonlinear_pde(N, dt, t_max):
    dz = H / (N - 1)
    b = np.full(N, b_T)
    b[0] = b_B
    b[-1] = b_T
    t = 0.0
    while t < t_max:
        b_new = b.copy()
        for j in range(1, N - 1):
            conv = alpha * ((b[j + 1]**3 - b[j - 1]**3) / (2 * dz))
            b_jph = 0.5 * (b[j] + b[j + 1])
            b_jmh = 0.5 * (b[j - 1] + b[j])
            db_jph = (b[j + 1] - b[j]) / dz
            db_jmh = (b[j] - b[j - 1]) / dz
            flux_ph = beta * b_jph**3 * db_jph
            flux_mh = beta * b_jmh**3 * db_jmh
            diff = (flux_ph - flux_mh) / dz
            b_new[j] = b[j] - dt * (conv - diff)
        b_new = np.nan_to_num(b_new, nan=b_T, posinf=b_T, neginf=b_T)
        b_new[0] = b_B
        b_new[-1] = b_T
        b = b_new
        t = round(t + dt, 10)
    return b


def compute_l2_norm(error, dz):
    return np.sqrt(np.trapezoid(error**2, dx=dz))


grid_sizes = [11, 21, 41]
grid_spacings = []
l2_errors = []

for N in grid_sizes:
    dz = H / (N - 1)
    grid_spacings.append(dz)
    b_numerical = solve_nonlinear_pde(N, dt, t_max)
    b_steady = compute_steady_state(N)
    error = b_steady - b_numerical
    l2 = compute_l2_norm(error, dz)
    l2_errors.append(l2)


grid_spacings = np.array(grid_spacings)
l2_errors = np.array(l2_errors)


log_h = np.log(grid_spacings)
log_error = np.log(l2_errors)
slope, intercept, _, _, _ = linregress(log_h, log_error)
fit_line = np.exp(intercept) * grid_spacings**slope

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(grid_spacings, l2_errors, marker='o', label='L2 Error')
plt.plot(grid_spacings, fit_line, linestyle='--', label=f'Fit: slope = {slope:.2f}')
plt.xlabel('Grid Spacing (Δz)')
plt.ylabel('L2 Norm of Error')
plt.title('L2 Error vs Grid Spacing (log-log plot)')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
