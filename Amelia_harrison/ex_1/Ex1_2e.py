# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:57:07 2025

@author: qlcq0853
numerics last part
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import time as tm

def residual(b_new, b_old, delta_t, delta_z, alpha, beta,j):
    fc_new = np.zeros(j)
    fc_old = np.zeros(j)
    fd_new = np.zeros(j)
    fd_old = np.zeros(j)
    conv = np.zeros(j)
    diffu = np.zeros(j)
    R = np.zeros(j)
    
    for k in range(0,j-1):
        #Initial values of convective, diffusive flux
        fc_new[k] = alpha*b_new[k]**3 #at n+1
        fc_old[k] = alpha*b_old[k]**3 #at n
        
        fd_new[k] = beta*((b_new[k+1] + b_new[k])**3
                          *(b_new[k+1]-b_new[k])
                          )
        
        fd_old[k] = beta*((b_old[k+1] + b_old[k])**3
                          *(b_old[k+1]-b_old[k])
                          )
    for k in range(1,j-1):
        conv[k] = (fc_new[k]-fc_new[k-1]
                +fc_old[k]-fc_old[k-1])/(2*delta_z)
        
        diffu[k] = (fd_new[k]-fd_new[k-1]
                 +fd_old[k]-fd_old[k-1])/(16*delta_z**2)
    
    for k in range(1, j-1):
        
        #now defining residual
        
        R[k] = b_new[k]-b_old[k] +delta_t*(conv[k]-
                                           diffu[k])
    return R

#timing code
start = tm.time()
j = 11
time = [ 1, 2, 3, 4, 5]


#value of constants
Q=0.99
H=3
alpha=0.4709
beta = 1.0
delta_z = H/(j-1)
#note, max possible b_b = 1 due to arctanh
t_max = (1)*(1/(alpha/delta_z +(2*beta)/delta_z**2))

n = math.ceil(1/t_max)+1
delta_t = 1/(n-1)

b_old = np.zeros(j)
b = np.linspace(0, 1, 10000)
num_res = np.zeros([len(time),j])
exact_res = np.zeros([len(time), len(b)])
z_ref = 0.9

#initial conditions
z_init = z_ref +(1/alpha)*(b-np.arctanh(b))
b_interp = interp1d(b,z_init,bounds_error=False, fill_value="extrapolate")

#initial conditons based on intial exact sol
b_it = 0

b_old[:] = b_it
z_count = np.floor(0.9/delta_z)
#creating inital conditions which match exact sol at t0
z_grid =np.linspace(0,0.9,int(z_count))
b_init = b_interp(z_grid)
b_old[:int(z_count)] = b_init

b_new_init = b_old.copy()
max_iter = 10
for t in range(len(time)):
    #exact solution
    z = z_ref + alpha*time[t] +(1/alpha)*(b-np.arctanh(b))
    exact_res[t,:] = z
    #numerical solution
    #reseting b to val at t=0 for each time
    b_new = b_new_init.copy()
    b_old = b_new_init.copy()
    #calculate timestep
    n = math.ceil(time[t]/t_max)+1
    delta_t =time[t]/(n-1)
    for a in range(1,n):
        #setting BCs at t = a*delta_t
        z_val = z_ref + alpha*a*delta_t +(1/alpha)*(b-np.arctanh(b))
        b_exact = b[(z_val>= 0) & (z_val<=3)]
        b_t = np.min(b_exact)
        b_b = np.max(b_exact)
        b_new[0] = b_b
        b_new[j-1] = b_t
        b_old[0] = b_b
        b_old[j-1] = b_t
        for k in range(max_iter):
            R = residual(b_new, b_old, delta_t,delta_z,alpha,beta,j)
            #jacobian calculation 
            #numerically, using perturbations and residuals
            J = np.zeros((j,j))
            eps = 1e-8
            
            for l in range(1,j-1):
                b_pert = b_new.copy()
                #perturbing b at b_l
                b_pert[l] += eps
                #effect of perturbation on residual
                R_pert = residual(b_pert, b_old, delta_t,delta_z,alpha,beta,j)
                J[:, l] = (R_pert[:] - R[:]) / eps
                
            #solving J*delta_b = -R
            delta = np.linalg.solve(J[1:j-1,1:j-1], -R[1:j-1])
            if np.linalg.norm(delta)<1e-10:
                break
            b_new[1:j-1] += delta
        #print(b_new[1])
        b_old = b_new.copy()
            #need to set new values of b_b, b_t
    num_res[t,:] = b_new
    
z = np.linspace(0, 3, j)
for k in range(len(time)):
    
    
    plt.plot(b/2, exact_res[k,:], c=f'C{k}', ls='--')
    plt.plot(-b/2, exact_res[k,:], c=f'C{k}', ls='--')
    
    plt.plot(num_res[k,:]/2, z, c=f'C{k}', label=f't={time[k]}s')
    plt.plot(-num_res[k,:]/2, z, c=f'C{k}')
#plt.plot(b_new/2,z)
#plt.plot(b/2, exact_res[0,:])
plt.legend(bbox_to_anchor=(1,1))   
plt.ylim(0,3)
plt.xlim(-0.5, 0.5)
plt.xlabel('Dike width')
plt.ylabel('z')
plt.title('Dike width at different times (j=11)')

end = tm.time()
print(end-start)



