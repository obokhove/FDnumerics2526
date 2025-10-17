# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 20:56:13 2025

@author: wfrt0938
"""
import numpy as np
import matplotlib.pyplot as plt
from steadystate import steadystate
from nonlinearsolver import nonlinearsolver

J=41
Z= np.linspace(0,1,J)
#Generate results 
[B_ss,Z_ss]=steadystate(J)  #steady state solution

B1=nonlinearsolver(J,0.05)
B2=nonlinearsolver(J,0.1)
B3=nonlinearsolver(J,0.2)
B4=nonlinearsolver(J,0.5)
B5=nonlinearsolver(J,1)
B6=nonlinearsolver(J,2)


halfb_ss=B_ss/2
plt.plot(halfb_ss,Z_ss, color='blue',label='Steady state solution')
plt.plot((-1*halfb_ss),Z_ss, color='blue')
plt.title('Height vs. Dike Width for J=41' )
plt.xlabel('Dike Wall Position (-b/2 to + b/2)')
plt.ylabel('Height (z)')

halfb1=B1/2
plt.plot(halfb1,Z, color='red',label='t=0.05')
plt.plot((-1*halfb1),Z, color='red')

halfb2=B2/2
plt.plot(halfb2,Z, color='orange',label='t=0.1')
plt.plot((-1*halfb2),Z, color='orange')

halfb3=B3/2
plt.plot(halfb3,Z, color='yellow',label='t=0.2')
plt.plot((-1*halfb3),Z, color='yellow')


halfb4=B4/2
plt.plot(halfb4,Z, color='purple',label='t=0.5')
plt.plot((-1*halfb4),Z, color='purple')

halfb5=B5/2
plt.plot(halfb5,Z, color='green',label='t=1')
plt.plot((-1*halfb5),Z, color='green')

halfb6=B6/2
plt.plot(halfb6,Z, color='grey',label='t=2')
plt.plot((-1*halfb6),Z, color='grey')
plt.legend()
