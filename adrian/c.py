# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:53:06 2025

@author: xbfl0349
"""
import numpy as np
import matplotlib.pyplot as plt

Q = 0.99
alpha = 0.4709
beta = 1.0
H = 1.0
b_B = 1.178164343


"""---- Temporal Solution Parameters ----"""
t_step = 0.00005
b_T =  0.585373798
t_terminal = 2

def steady_state(Q, alpha, beta, H, b_B, N_points):
    
    y = [b_B]
    delt_z = H / (N_points - 1)
    for i in range(N_points - 1):
        new = delt_z * (alpha - Q / ( y[i]**3)) / beta + y[i]
        y.append(new)
        
    return np.array(y)

def temporal_sol(Q, alpha, beta, H, b_B, b_T, N_points, t_step, terminal_time):
    def p(index, b):
        return (b[index]**3 + b[index - 1]**3) / 2
    
    delt_z = H / (N_points - 1)
    b = np.array([b_T for i in range(N_points)])
    b[0] = b_B
    t = 0.0
    while t <= terminal_time:
        
        #print('start')
        for i in range(1, N_points - 1):
            convection = - 3 * alpha * b[i]**2 * (b[i]-b[i-1]) * t_step / delt_z
            diffusion = beta * t_step / delt_z**2 * (p(i + 1, b) * (b[i+1] - b[i]) - p(i, b) * (b[i] - b[i-1]))
            b[i] = convection + diffusion + b[i]
            #print(str(b[i]))
        if True in np.isnan(b):
            raise ValueError("ERROR: Solution diverges! Refine time step!")
        '''--------- ROUNDING -------------'''
        t = round(t + t_step, 6)
    return b

def err_squared(b_steady, b_temp, N_points_std, N_points, index_temp):
    if (N_points_std - 1) % (N_points - 1) !=0:
        print("Non-overlaping z-domains!!! Change N_points of steady, or temporal solution so they have non-empty intersection!")
        return None
    
    index_factor = (N_points_std - 1) // (N_points - 1)
    return (b_steady[index_temp * index_factor] - b_temp[index_temp])**2

def L_2err(H, N_points_std, N_points, b_steady, b_temp):
    if (N_points_std - 1) % (N_points - 1) !=0:
        print("Non-overlaping z-domains!!! Change N_points of steady, or temporal solution so they have non-empty intersection!")
        return None
    
    errors = np.array([err_squared(b_steady, b_temp, N_points_std, N_points,  i) for i in range(N_points)])
    print(np.max(errors))
    
    """Remember first and last term are ommited!"""
    trap_sum = np.sum(errors[1:-1])
    delt_z = H / (N_points - 1)
    
    L_2 = np.sqrt(delt_z / 2 * (errors[0] + 2 * trap_sum + errors[N_points-1]))
    return L_2
    

# Solve for these spatial discretizations!!
N_points = np.array([5, 11, 21, 41, 101])
err = []
N_points_std = -1
N_points_std = 10001
b_steady = steady_state(Q, alpha, beta, H, b_B, N_points_std)
for N in N_points:
    """---- Calculate steady solution only if the two grids wont overlap! ----"""
    if (N_points_std - 1) % (N - 1) != 0:
        raise ValueError("ERROR: steady state solution grid points and temporal solution grid points do not overlap! Ensure that N_points_std - 1 is divisible by N - 1, for N=" + str(N))
    b_temp = temporal_sol(Q, alpha, beta, H, b_B, b_T, N, t_step, t_terminal)
    err.append(L_2err(H, N_points_std, N, b_steady, b_temp))

plt.plot(np.log2(N_points), np.log2(err))
for i in range(len(N_points)):
    plt.plot(np.log2(N_points[i]), np.log2(err[i]), 'o', label = str(N_points[i]) + " grid points, L2 error " + str(err[i]))
plt.xlabel("Logarithm of the number of grid points")
plt.ylabel("Logarithm of L2 error")
plt.title("L2 error of steady state vs temporal solution at t=2s")
plt.legend()
plt.show()
#err(delt_z_std, delt_z, N_points_std, N_points, b_steady, b_temp)