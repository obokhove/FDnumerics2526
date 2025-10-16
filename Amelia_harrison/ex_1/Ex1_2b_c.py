# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 13:22:06 2025

@author: qlcq0853
numerics ex 1 part 2
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import Ex1_2a
#no of grid in z direction
j = 41
time = [0.05, 0.1, 0.2, 0.5, 1, 2]

#value of constants
Q=0.99
H=1
alpha=0.4709
beta = 1.0
b_b = 1.178164343
b_t = 0.581402
delta_z = H/(j-1)
results = np.zeros([6,j])

for x in range(len(time)):

    #max time step
    t_max = (1/b_b**2)*(1/(alpha/delta_z +(2*beta*b_b)/delta_z**2))
    n = math.ceil(time[x]/t_max)+1
    delta_t =time[x]/(n-1)
    
    b_sol = np.zeros([n,j])
    
    #collecting terms
    c_1 = alpha*(delta_t/delta_z)
    c_2 =(beta*delta_t)/(8*delta_z**2)
    print(c_1, c_2)
    
    #initial conditions
    
    for k in range(0,j):
        b_sol[0,k] = b_t
        b_sol[0,0] = b_b
        
    #iterate through time
    
    for a in range(1,n):
        #setting j=0

        b_sol[a,0] = b_b
        
        #z=H (j-1)

        b_sol[a,j-1] = b_t
        # all other values of j
        for k in range(1, j-1):
            b_sol[a, k] = (b_sol[a-1,k] -c_1*(b_sol[a-1,k]**3 - b_sol[a-1, k-1]**3)
                           +c_2*((b_sol[a-1,k+1] + b_sol[a-1,k])**3*(b_sol[a-1, k+1] - b_sol[a-1,k])
                                 - (b_sol[a-1,k]+b_sol[a-1,k-1])**3*(b_sol[a-1,k]-b_sol[a-1,k-1]))
                           )
            
            
    results[x,:] = b_sol[n-1,:]

z = np.linspace(0,1,j)

#t=0.05
plt.plot(results[0,:]/2,z,c='C0', label='t=0.05s')
plt.plot(-results[0,:]/2,z,c='C0')
#t=0.1
plt.plot(results[1,:]/2,z,c='C1', label='t=0.1s')
plt.plot(-results[1,:]/2,z,c='C1')
#t=0.2
plt.plot(results[2,:]/2,z,c='C2', label='t=0.2s')
plt.plot(-results[2,:]/2,z,c='C2')
#t=0.5
#dots and dashes as t>0.5 all overlap
plt.plot(results[3,:]/2,z,c='C3',ls='--', label='t=0.5s')
plt.plot(-results[3,:]/2,z,c='C3',ls='--')
#t=1
plt.plot(results[4,:]/2,z,c='C4',ls='-.', label='t=1s')
plt.plot(-results[4,:]/2,z,c='C4',ls='-.')
#t=2
plt.plot(results[5,:]/2,z,c='C8',ls=':', label='t=2s')
plt.plot(-results[5,:]/2,z,c='C8',ls=':')

b_steady, z_s = Ex1_2a.steady_state()
plt.plot(b_steady/2,z_s,c='C9',alpha=0.7, label='steady state solution')
plt.plot(-b_steady/2,z_s, c='C9',alpha=0.7)
plt.legend(bbox_to_anchor=(1,1))

plt.xlim(-b_b/1.9, b_b/1.9)
plt.ylim(0, H)
plt.xlabel('Dike width')
plt.ylabel('z')
plt.title('Dike width at different times (j=41)')

#L2 calculation and e calc

e=np.zeros(j)
#making steady state have same delta_z as numerical
b_steady_interp = np.interp(z, z_s, b_steady)

#e calculation
e = b_steady_interp - results[5,:]

sum_L = e[0]**2 + e[j-1]**2
#looping over other vals as added twice
for i in range(1,j-1):
    sum_L = sum_L + 2*e[i]**2
#L2 calc
L_2 = np.sqrt((delta_z/2)*sum_L)
print(L_2)
#L_inf calc
L_inf = np.max(np.abs(e))
print(L_inf)


      