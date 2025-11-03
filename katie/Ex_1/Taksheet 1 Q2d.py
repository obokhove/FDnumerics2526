
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.4709
beta = 1.0
c = alpha
H = 3.0  # Total depth in km
zr0 = 0.9  # Dike closed below 2.1 km
b_max = 1.0
j_values = [11, 21, 41]
time_values = [1, 2, 3, 4, 5]

# Exact travelling wave solution
def exact_solution(b, t):
    return zr0 + c * t + (1 / alpha) * (b - np.arctanh(b))

# Maximum stable time step from maximum principle
def compute_max_dt(alpha, beta, dz):
    return 1 / (alpha / dz + 2 * beta / dz**2)


for j in j_values:
    dz = H / (j - 1)
    dt = compute_max_dt(alpha, beta, dz)
    z = np.linspace(0, H, j)
    num_res = np.zeros((len(time_values), j))
    b = np.linspace(0, 1, 10000)
    exact_res = np.zeros((len(time_values), len(b)))

    for t_idx, t_val in enumerate(time_values):
        # Exact solution
        z_exact = exact_solution(b, t_val)
        exact_res[t_idx, :] = z_exact

        # Numerical setup
        n_steps = int(np.ceil(t_val / dt)) + 1
        delta_t = t_val / (n_steps - 1)
        b_sol = np.zeros((n_steps, j))

        # Initial condition
        z_init = exact_solution(b, 0)
        b_interp = np.interp(np.linspace(0, zr0, int(np.floor(zr0 / dz))), z_init, b)
        b_sol[0, :len(b_interp)] = b_interp

        # Coefficients
        c1 = alpha * (delta_t / dz)
        c2 = beta * delta_t / (8 * dz**2)

        # Time stepping
        for n in range(1, n_steps):
            z_val = exact_solution(b, n * delta_t)
            b_valid = b[(z_val >= 0) & (z_val <= H)]
            b_b = np.max(b_valid)
            b_t = np.min(b_valid)
            b_sol[n, 0] = b_b
            b_sol[n, -1] = b_t
            for k in range(1, j - 1):
                b_sol[n, k] = (
                    b_sol[n - 1, k]
                    - c1 * (b_sol[n - 1, k]**3 - b_sol[n - 1, k - 1]**3)
                    + c2 * (
                        (b_sol[n - 1, k + 1] + b_sol[n - 1, k])**3 * (b_sol[n - 1, k + 1] - b_sol[n - 1, k])
                        - (b_sol[n - 1, k] + b_sol[n - 1, k - 1])**3 * (b_sol[n - 1, k] - b_sol[n - 1, k - 1])
                    )
                )
        num_res[t_idx, :] = b_sol[-1, :]

    # Plotting
    plt.figure(figsize=(6, 8))
    for t_idx, t_val in enumerate(time_values):
        color = f'C{t_idx}'
        plt.plot(num_res[t_idx, :] / 2, z, color=color, label=f't={t_val}')
        plt.plot(-num_res[t_idx, :] / 2, z, color=color)
        plt.plot(b / 2, exact_res[t_idx, :], linestyle='--', color=color)
        plt.plot(-b / 2, exact_res[t_idx, :], linestyle='--', color=color)
    plt.xlabel('Dike width')
    plt.ylabel('Depth z (km)')
    plt.title(f'Dike width at different times (j={j})')
    plt.legend(loc='upper right')
    plt.ylim(0, 5)
    plt.xlim(-0.5, 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'dike_profile_implicit_j{j}.png')
    plt.show()
