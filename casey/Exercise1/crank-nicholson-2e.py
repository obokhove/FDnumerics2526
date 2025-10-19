# -*- coding: utf-8 -*-
"""
Crank-Nicholson method
"""
#2e

import numpy as np
import matplotlib.pyplot as plt
import time as tm

def residual(b_new, b_old, dt, dz, alpha, beta, j): # residual function for b
    cf_new = np.zeros(j)
    cf_old = np.zeros(j)
    df_new = np.zeros(j)
    df_old = np.zeros(j)
    conv = np.zeros(j)
    dif = np.zeros(j)
    R = np.zeros(j)
    
    for k in range(0,j-1):
        # Initial values of convective, diffusive flux
        cf_new[k] = alpha*b_new[k]**3 #at n+1
        cf_old[k] = alpha*b_old[k]**3 #at n
        
        df_new[k] = beta*((b_new[k+1] + b_new[k])**3
                          *(b_new[k+1]-b_new[k]))
        
        df_old[k] = beta*((b_old[k+1] + b_old[k])**3
                          *(b_old[k+1]-b_old[k]))
        
    for k in range(1,j-1):
        conv[k] = (cf_new[k] - cf_new[k-1]
                + cf_old[k] - cf_old[k-1])/(2*dz)
        
        dif[k] = (df_new[k] - df_new[k-1]
                 + df_old[k] - df_old[k-1])/(16*dz**2)
    
    for k in range(1, j-1):       
        # defining residual
        R[k] = b_new[k]-b_old[k] +dt*(conv[k] - dif[k])
        
    return R

# timing code
start = tm.time()
# grid points in z direction
j = 41
time = np.array([1, 2, 3, 4, 5])

#-------------------------------
# value of constants
#-------------------------------
Q = 0.99
H = 3
alpha = 0.4709
beta = 1.0
dz = H/(j-1)
#note, max possible bB = 1 due to arctanh
t_max = (1)*(1/(alpha/dz + (2*beta)/dz**2))
#-------------------------------
#-------------------------------

#-------------------------------
# Iterative method
#-------------------------------
b_old = np.zeros(j)
b = np.linspace(0, 1, 10000)
num_res = np.zeros([len(time),j])
exact_res = np.zeros([len(time), len(b)])
z_ref = 0.9

# initial conditions
z_init = z_ref +(1/alpha)*(b-np.arctanh(b))


# initial conditons based on intial exact solution
b_it = 0

b_old[:] = b_it
z_count = np.floor(0.9/dz)
# creating inital conditions which match exact solution at t0
z_grid =np.linspace(0,0.9,int(z_count))
b_init = np.interp(z_grid, z_init, b)
b_old[:int(z_count)] = b_init

b_new_init = b_old.copy()
max_iter = 10
for t in range(len(time)):
    # exact solution
    z = z_ref + alpha*time[t] +(1/alpha)*(b-np.arctanh(b))
    exact_res[t,:] = z
    # numerical solution
    # resetting b val at t=0 for each time
    b_new = b_new_init.copy()
    b_old = b_new_init.copy()
    # calculate timestep
    n = np.ceil(time[t]/t_max).astype(int) + 1
    dt = time[t]/(n-1)

    for a in range(1,n):
        # setting BCs at t = a*dt
        z_val = z_ref + alpha*a*dt + (1/alpha)*(b-np.arctanh(b))
        b_exact = b[(z_val>= 0) & (z_val<=3)]
        bT = np.min(b_exact)
        bB = np.max(b_exact)
        b_new[0] = bB
        b_new[j-1] = bT
        b_old[0] = bB
        b_old[j-1] = bT
        for k in range(max_iter):
            R = residual(b_new, b_old, dt, dz, alpha, beta, j)
            # jacobian calculation 
            # numerically, using perturbations and residuals
            J = np.zeros((j,j))
            epsilon = 1e-8
            
            for l in range(1,j-1):
                b_pert = b_new.copy()
                #perturbing b at b_l
                b_pert[l] += epsilon
                #effect of perturbation on residual
                R_pert = residual(b_pert, b_old, dt, dz, alpha, beta, j)
                J[:, l] = (R_pert[:] - R[:]) / epsilon
                
            # solving J*db = -R
            db = np.linalg.solve(J[1:j-1,1:j-1], -R[1:j-1])
            if np.linalg.norm(db)<1e-10:
                break
            b_new[1:j-1] += db
        b_old = b_new.copy()
            # set new values of bB, bT
    num_res[t,:] = b_new
#-------------------------------
#-------------------------------

#-------------------------------
# plotting
#-------------------------------
z = np.linspace(0,3,j)

for k in range(len(time)):
    plt.plot(b/2, exact_res[k,:], c=f'C{k}', ls='--')
    plt.plot(-b/2, exact_res[k,:], c=f'C{k}', ls='--')

    plt.plot(num_res[k,:]/2, z, c=f'C{k}', label=f't = {time[k]}s')
    plt.plot(-num_res[k,:]/2, z, c=f'C{k}')

plt.legend()
plt.ylim(0,H)
plt.xlim(-0.5, 0.5)
plt.xlabel('Dike width (m)')
plt.ylabel('z(m)')
plt.title(f'Time progression of Dike width (j = {j}) C-N')
# time at end
end = tm.time()
print(end-start)