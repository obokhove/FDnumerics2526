# -*- coding: utf-8 -*-
"""
Implicit exact solution
"""
#2d

import numpy as np
import matplotlib.pyplot as plt
import time as tm

# timing code
start = tm.time()
# grid points in z direction
j = 11
time = np.array([1, 2, 3, 4, 5])

#-------------------------------
# value of constants
#-------------------------------
Q = 0.99
H = 3
alpha = 0.4709
beta = 1.0
bB = 1.178164343
bT = 0.581402
dz = H/(j-1)
#note, max possible bB = 1 due to arctanh
t_max = (1)*(1/(alpha/dz + (2*beta)/dz**2))
#-------------------------------
#-------------------------------

#-------------------------------
# Iterative method
#-------------------------------
num_res = np.zeros([len(time), j])
b = np.linspace(0, 1, 10000)
exact_res = np.zeros([len(time), len(b)])

z_ref = 0.9
b = np.linspace(0, 1, 10000)
for t in range(len(time)):
    # exact solution
    z = z_ref + alpha*time[t] + (1/alpha)*(b-np.arctanh(b))
    exact_res[t,:] = z

    # numerical solution
    # calculate timestep
    n = np.ceil(time[t]/t_max).astype(int) + 1
    dt = time[t]/(n-1)

    # initial conditions
    z_init = z_ref + (1/alpha)*(b-np.arctanh(b))

    b_it = 0

    b_sol = np.zeros([n,j])

    # finding z where b is non zero
    z_count = np.floor(0.9/dz).astype(int) # Used dz

    # creating inital conditions which match exact solution at t0
    z_grid_points = np.linspace(0, H, j)
    z_interp_target = z_grid_points[:z_count]

    b_init = np.interp(z_interp_target, z_init, b)

    b_sol[0,:z_count] = b_init

    # collecting terms
    c_1 = alpha*(dt/dz)
    c_2 =(beta*dt)/(8*dz**2)


    # iterate through time
    for a in range(1,n):
        z_val = z_ref + alpha*a*dt +(1/alpha)*(b-np.arctanh(b)) # Used dt
        b_exact = b[(z_val>= 0) & (z_val<=3)]
        bT_val = np.min(b_exact)
        bB_val = np.max(b_exact)

        # j=0 (Boundary condition at the bottom)
        b_sol[a,0]= bB_val
        # j-1 (Boundary condition at the top)
        b_sol[a,j-1] = bT_val

        #other vals
        for k in range(1, j-1):
            b_sol[a, k] = (b_sol[a-1,k] -c_1*(b_sol[a-1,k]**3 - b_sol[a-1, k-1]**3)
                            +c_2*((b_sol[a-1,k+1] + b_sol[a-1,k])**3*(b_sol[a-1, k+1] - b_sol[a-1,k])
                                    - (b_sol[a-1,k]+b_sol[a-1,k-1])**3*(b_sol[a-1,k]-b_sol[a-1,k-1])))
    num_res[t,:] = b_sol[n-1,:]
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
plt.title(f'Time progression of Dike width (j = {j})')
# time at end
end = tm.time()
print(end-start)