# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:35:32 2025

@author: qlcq0853
numerics ex1 implicit solution
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import time as tm
#timing code
start = tm.time()
#no of grid in z direction
j = 41
time = [ 1, 2, 3, 4, 5]

#value of constants
Q=0.99
H=3
alpha=0.4709
beta = 1.0
b_b = 1.178164343
b_t = 0.581402
delta_z = H/(j-1)
#note, max possible b_b = 1 due to arctanh
t_max = (1)*(1/(alpha/delta_z +(2*beta)/delta_z**2))

num_res = np.zeros([len(time),j])
b = np.linspace(0, 1, 10000)
exact_res = np.zeros([len(time), len(b)])

z_ref = 0.9
b = np.linspace(0, 1, 10000)
for t in range(len(time)):
    #exact solution
    z = z_ref + alpha*time[t] +(1/alpha)*(b-np.arctanh(b))
    exact_res[t,:] = z
    #plt.plot(b/2, z)
    #plt.plot(-b/2, z)
    #numerical solution
    #calculate timestep
    n = math.ceil(time[t]/t_max)+1
    delta_t =time[t]/(n-1)
    
    #initial conditions
    z_init = z_ref +(1/alpha)*(b-np.arctanh(b))
    #z_ex = z_init[(z_init>=0.0)& (z_init<=3)]
    #b_ex = b[(z_init>=0.0)& (z_init<=3)]
    b_interp = interp1d(b,z_init,bounds_error=False, fill_value="extrapolate")
    #z_grid = (0,3,j)
    #b_init = b_interp(z_grid)
    
    b_it = 0
    
    b_sol = np.zeros([n,j])
    b_sol[0,:] = b_it
    
    #finding at which z, b is non zero
    z_count = np.floor(0.9/delta_z)
    #creating inital conditions which match exact sol at t0
    z_grid =np.linspace(0,0.9,int(z_count))
    b_init = b_interp(z_grid)
    b_sol[0,:int(z_count)] = b_init
    #collecting terms
    c_1 = alpha*(delta_t/delta_z)
    c_2 =(beta*delta_t)/(8*delta_z**2)
    
        
    #iterate through time
    for a in range(1,n):
        z_val = z_ref + alpha*a*delta_t +(1/alpha)*(b-np.arctanh(b))
        b_exact = b[(z_val>= 0) & (z_val<=3)]
        b_t = np.min(b_exact)
        b_b = np.max(b_exact)
        #j=0
        b_sol[a,0]= b_b
        #j-1
        b_sol[a,j-1] = b_t
        
        #other vals
        for k in range(1, j-1):

                
            b_sol[a, k] = (b_sol[a-1,k] -c_1*(b_sol[a-1,k]**3 - b_sol[a-1, k-1]**3)
                               +c_2*((b_sol[a-1,k+1] + b_sol[a-1,k])**3*(b_sol[a-1, k+1] - b_sol[a-1,k])
                                     - (b_sol[a-1,k]+b_sol[a-1,k-1])**3*(b_sol[a-1,k]-b_sol[a-1,k-1]))
                               )
            #else:
                #b_sol[a,k] = b_t
    num_res[t,:] = b_sol[n-1,:]
    #plt.plot(b_sol[n-1,:]/2, z)
    #results[t,:] = b_sol[n-1,:]

#plotting
z = np.linspace(0,3,j)

for k in range(len(time)):
    
    
    plt.plot(b/2, exact_res[k,:], c=f'C{k}', ls='--')
    plt.plot(-b/2, exact_res[k,:], c=f'C{k}', ls='--')
    
    plt.plot(num_res[k,:]/2, z, c=f'C{k}', label=f't={time[k]}s')
    plt.plot(-num_res[k,:]/2, z, c=f'C{k}')
   
plt.legend(bbox_to_anchor=(1,1))   
plt.ylim(0,3)
plt.xlim(-0.5, 0.5)
plt.xlabel('Dike width')
plt.ylabel('z')
plt.title('Dike width at different times (j=41)')
#time a end f progrm
end =tm.time()
print(end-start)
#print(b[z==0])
