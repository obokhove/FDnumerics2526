#######################################################################
#            Exercise-example 1, numerics for PDE's Part I
#            within Fluid Dynamcis module MATH5453 2025ex
#               (O. Bokhove: o.bokhove@leeds.ac.uk)
#            Solution to:
#            lamb_t = lamba_xx-int_Lp^L lamb(xtilde,t) d xtilde-f(x)
#######################################################################


##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import matplotlib.pyplot as plt
import time

##################################################################
# Define parameters
##################################################################
L = 4
Lp = 2
Nx = 5
Nm = 4
Linf = np.linspace(1, Nm, Nm) 
L2 = np.linspace(1, Nm, Nm) 
dxn = np.linspace(1, Nm, Nm) 
NcNx = np.linspace(1, Nm, Nm) 
print("Linf dxn",Linf,dxn)
plt.ion()
for nnx in range(1, Nm+1):
    dx = (L-Lp)/(1.0*Nx)
    xj = np.linspace(Lp, L, Nx+1) 
    lamb0 = 0.0
    lamb = 0.0*xj
    lambn = 0.0*lamb;
    lambs = (xj-Lp)*(xj-2.0*L+Lp)
    Tend = 14.0 #
    tijd = 0.0
    Nt = 40
    dtmeet = Tend/(1.0*Nt)
    tmeet = dtmeet
    #
    # Initial condition
    #
    lamb = ((xj-Lp)**2)*(xj-2*L+Lp)**2
    ff = 2.0+(2.0/3.0)*(L-Lp)**3
    CFL = 0.6
    #
    # Plot intial condition
    #    plt.figure(1)
    ax1 = plt.subplot(221)
    ax1.plot(xj,lamb,'-k',lw=2)
    ax1.plot(xj,lambs,'--r',lw=2)
    plt.xlabel('$x$',fontsize=18)
    plt.ylabel('$\lambda(x,t)$',fontsize=18)
    plt.axis([2, 4, -5, 20])
    plt.pause(0.05)
    #
    # Enter time loop
    #
    while tijd <= Tend:
        dt = CFL*0.5*dx**2 # Time step estimate based on diffusion
        mu = dt/dx**2
        tijd = tijd+dt
        # ??? Comment out discretization:
        intlamb = dx*( 0.5*(lamb0+lamb[Nx]) + sum(lamb[1:Nx]) ) # Note: one extra point!
        lambn[1] = lamb[1]+mu*(lamb0-2.0*lamb[1]+lamb[2])-dt*intlamb-dt*ff
        lambn[2:Nx] = lamb[2:Nx]+mu*(lamb[1:Nx-1]-2.0*lamb[2:Nx]+lamb[3:Nx+1])-dt*intlamb-dt*ff # Note: one extra point!
        lambn[Nx] = lamb[Nx]+mu*(lamb[Nx-1]-2.0*lamb[Nx]+lamb[Nx-1])-dt*intlamb-dt*ff
        lamb[0:Nx+1] = lambn[0:Nx+1]
        # ??? End comment out.
        #
        # Plot solution at set times
        #
        if tijd>=tmeet:
            tmeet = tmeet+dtmeet
            plt.figure(1)
            ax1.plot(xj,lamb,'-k',lw=2)
            ax1.plot(xj,lambs,'--r',lw=2)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$\lambda(x,t)$',fontsize=18)
            plt.axis([2, 4, -5, 20]) # Does not work in python3
         # End plot.  
    # ??? Comment out making of Linf and L2 errors:
    Linf[nnx-1] = max(abs(lamb-lambs))
    L2[nnx-1] = np.sqrt( dx*0.5*((lamb[0]-lambs[0])**2+(lamb[Nx]-lambs[Nx])**2)+dx*sum( (lamb[1:Nx]-lambs[1:Nx])**2))/np.sqrt(L-Lp)
    dxn[nnx-1] = dx
    NcNx[nnx-1] = Nx
    Nx = 2*Nx
    # ??? End comment out.
    # End while loop.
    print("nnx:",nnx)

print(" ------------ END OF PROGRAM ----------")  
# 
# Plotting of Linfty and L2 errors and the respective expected slopes.
# 
plt.figure(1)
ax2 = plt.subplot(222)
print("dxn, Linf",dxn,Linf)
# ??? Comment out plotting of Linfty error and slope:
ax2.plot(np.log(dxn),np.log(Linf),'x',lw=2);
print("dxn0, Linf0",dxn,Linf)
print("dxn0, dxN, Linf[0] Inf[Nm-1]",dxn[0],dxn[Nm-1],Linf[0],Linf[Nm-1])
plt.plot([-3.0,-1.0],[-8,-4],'--k') 
ax2.plot([np.log(dxn[Nm-1]),np.log(dxn[0])],[np.log(Linf[Nm-1]),np.log(Linf[Nm-1])+2.0*(np.log(dxn[0])-np.log(dxn[Nm-1]))],'-b');
# ??? End comment out.
plt.xlabel('$ln(\Delta x)$',fontsize=18)
plt.ylabel('$ln(L^\infty)$',fontsize=18)
ax3 = plt.subplot(223)
ax3.plot(np.log(dxn),np.log(L2),'x',lw=2)
ax3.plot([np.log(dxn[Nm-1]),np.log(dxn[0])],[np.log(L2[Nm-1]),np.log(L2[Nm-1])+2.0*(np.log(dxn[0])-np.log(dxn[Nm-1]))],'-b')
plt.plot([-3.0,-1.0],[-8.4,-4.4],'--k') 
plt.xlabel('$ln(\Delta x)$',fontsize=18)
plt.ylabel('$ln(L_2)$',fontsize=18)

plt.show(block=True)
print(" ------------ END OF PROGRAM ----------")  

##################################################################    
#                        END OF PROGRAM                          #
##################################################################

