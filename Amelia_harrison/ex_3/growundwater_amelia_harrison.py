from firedrake import *
#
# packages Working code as of May 14th 2018 by Will Booker and Onno Bokhove
# import numpy as np
from math import pow
import time as tijd # OB2025
import numpy as np # OB2025
import matplotlib # OB2025
import matplotlib.pyplot as plt
import random


m  = 40
Ly = 0.85
dy = Ly/m
mesh = IntervalMesh(m, 0 , Ly)
yvals = np.linspace(0, Ly, m)
# OB2025 y = mesh.coordinates # OB2025 Mesh coordinates
y, = SpatialCoordinate(mesh) # OB2025

# Time definitions
t   = 0.0
end = 100
Ntm = 50
dtmeas = end/Ntm
tmeas = dtmeas
tlarge = dtlarge = 10

# choice to compare with finite diff method
# 0 if no, 1 if yes
finite_diff = 0


# Define Function space on our mesh.
# Initially we will use a continuous linear Lagrange basis
# Try other order, 1, 2, 3
nCG = 1 # OB2025
V = FunctionSpace(mesh, "CG", nCG) # OB2025

# Define timestep value
CFL = 2.3
Dt = CFL*0.5*dy*dy/(nCG)**2  # Based on FD estimate; note that dt must be defined before flux, etc
# Dt = 16*Dt
# dt.assign(CFL*0.5*dy*dy)

dt = Constant(Dt) # Using dt.assign in the while loop should avoid having to rebuild the solver iirc

# Define Crank Nicholson parameter
theta = 0.0

#choosing whether rain is variable
Rvar = 0
# 0 if constant, 1 if variable

# Define Groundwater constants
mpor  = 0.3
sigma = 0.8
Lc    = 0.05
kperm = 1e-8
w     = 0.1
R     = Constant(0.000125)
nu    = 1.0e-6
g     = 9.81
alpha = kperm/( nu * mpor * sigma )
gamma_pen = Constant(1e6)  # strong penalty to enforce h(L) ≈ h(L-dy)
gam   = Lc/( mpor*sigma )
fac2  = sqrt(g)/( mpor*sigma )
rain_options = [1.0, 2.0, 4.0, 9.0]
blocknow = -1
rain = 0.000125

#variables for finite diff
h_diff = np.zeros(m)
h_new = np.zeros(m)
hcm = 0.0

# Storage for finite diff
times = []
hcm_hist = [0]
profiles = {}
profiles[0] = h_diff.copy()
profiles_el = {}
profiles_el[0] = h_diff.copy()

#
# ncase = 0 Dirichlet bc, ncase = 1 overflow groundwater into canal section with weir equation:
nncase = 1

# Initial condition
# OB2025: h_prev = Function(V)
# OB2025 old stiff commented out: h_prev.interpolate(Expression("0.0"))
# OB2025
h_prev = Function(V).interpolate(0.0 + 0.0*y) # OB2025 IC, I guess hnum = 0.0*y h_prev.interpolate(hnum)
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_ylabel(r'$h_{m}(y,t)$ [m]')
ax1.set_xlabel(r'$y$ [m]')
ax2.set_ylabel(r'$h_{cm}(y,t)$ [m]')
ax2.set_xlabel(r'$t$ [s]')
ax3.set_ylabel(r'$R(t)$ [m/s]')
ax3.set_xlabel(r'$t$ [s]')
#ax1.set_xlim(0,Lc)
#ax1.set_ylim(-0.0005, 0.008)
ax2.set_xlim(0,100)
ax3.set_xlim(0,100)
# Create storage for paraview
outfile = VTKFile("./Results/groundwater_onnob.pvd")

# Write IC to file for paraview
outfile.write(h_prev , t = t )
time_array = np.array([0])
tr = np.array([0])
h_m = np.array([h_prev.at(y) for y in yvals])
h_cm = np.array([h_prev.at(0)])
R_arr = np.array(rain)
ax1.plot(yvals, h_m, label = 't = 0.00 s')
# Define trial and test functions on this function space
# h will be the equivalent to h^n+1 in our timestepping scheme

phi = TestFunction(V)

def flux ( h , phi , R ):  # phi is test function q in (31) and (32)
    return ( alpha * g * h * dot ( grad (h) , grad (phi) ) - (R * phi )/ ( mpor * sigma ) )

def Rval(t): #this function varies the rain
    global blocknow, rain_options, rain, Rvar
    R0 = 0.000125
    if Rvar == 0:
        rain = 10
        return R0
    elif Rvar == 1:
        block = int(t/10)

        if block != blocknow:
            blocknow = block
            rain = random.choice(rain_options)

        if (t - 10*block) < rain:
            return R0
        else:
            return 0

R.assign(Rval(t))
print('Rain falls for', rain, 's')

## NB: Linear solves use TrialFunctions, non-linear solves use Functions with initial guesses.

if nncase == 0:
   # Provide intial guess to non linear solve
   h = Function(V)
   h.assign(h_prev)
   F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx
   # Boundary conditions: Condition at Ly satisfied weakly
   bc1 = DirichletBC(V, 0.07, 1)
   h_problem = NonlinearVariationalProblem( F , h , bcs = bc1)

elif nncase == 1:
   if theta == 0.0: # Matches (31)
     h, out = TrialFunction(V), Function(V) # Has to be set for linear solver
     aa = (h*phi/dt)*dx+(gam*phi*h/dt)*ds(1)
     L2 = ( h_prev*phi/dt - flux ( h_prev, phi, R) ) *dx
     L = L2+( gam*phi*h_prev/dt-phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1) # Matches (29)
     explicit_problem = LinearVariationalProblem(aa, L, out)
     explicit_solver = LinearVariationalSolver(explicit_problem, solver_parameters={'mat_type':'aij',
        'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})
   elif theta > 0.0: # Matches (30) when theta=1/2
     h = Function(V)
     h.assign(h_prev)
     #F = dh/dt term + b_n+1 term + b_n term
     F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx
     # Add boundary contributions at y = 0:
     # F2 = Lc (for n, n+1) + max val terms (for n, n+1)
     F2 = ( gam*phi*(h-h_prev)/dt+theta*phi*fac2*np.power(max_value(2.0*h/3.0,0.0),1.5)+(1-theta)*phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1)
     #Solves F, F2 using a newton method
     h_problem = NonlinearVariationalProblem( F+F2 , h )
     h_solver = NonlinearVariationalSolver(h_problem, solver_parameters={'mat_type':'aij','ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})

# Time loop
while (t < end):
    # First we increase time
    t += Dt
    # Print to console the current time
    # Use the solver and then update values for next timestep
    R.assign(Rval(t))
    R_arr = np.append(R_arr, Rval(t))
    tr = np.append(tr, t)

    if finite_diff ==1:
        h_diff[0] = hcm
        h_diff[-1] = h_diff[-2]

        # Interior points
        for j in range(1, m-1):
            h_new[j] = (h_diff[j] + ((Dt * alpha * g) / (2 * dy ** 2)) * (
                        (h_diff[j + 1] + h_diff[j]) * (h_diff[j + 1] - h_diff[j]) - (h_diff[j] + h_diff[j - 1]) * (h_diff[j] - h_diff[j - 1]))
                        + Dt * Rval(t) / (mpor * sigma))

        #setting hcm using cannal eq
        hcm = (hcm + ((Dt * mpor * sigma * alpha * g) / (2 * Lc)) * (h_diff[1]**2 - h_diff[0]**2) / dy
               - (np.sqrt(g) * Dt / Lc) * (np.max([(2/3)*h_diff[0], 0]) ** (3 / 2)))

        h_new[0] = hcm
        h_new[-1] = h_new[-2]
        # Update
        h_diff[:] = h_new[:]

    if theta == 0.0:
           explicit_solver.solve()
           h_prev.assign(out) # has to be renamed via out

    elif theta > 0.0:
           h_solver.solve()
           h_prev.assign(h)
        # Write output to file for paraview visualisation
    if t>tmeas:
        #getting hcm every 2s
        tmeas = tmeas+dtmeas
        outfile.write(h_prev , t = t )
        #h_m = np.array([h_prev.at(y) for y in yvals])
        #ax1.plot(yvals, h_m)
        time_array = np.append(time_array,t)
        h_cm = np.append(h_cm, [h_prev.at(0)])
        if finite_diff ==1:
            hcm_hist = np.append(hcm_hist, hcm)

    if t>tlarge:
        #plotting hm every 10s
        print('Time is: ', t)
        print('Rain falls for', rain, 's')
        tlarge = tlarge + dtlarge
        h_m = np.array([h_prev.at(y) for y in yvals])
        ax1.plot(yvals, h_m, label=f"t = {t:.2f} s")
        if finite_diff ==1:
            profiles[t] = h_diff.copy()
            profiles_el[t] = h_m.copy()


# End while time loop
ax2.plot(time_array, h_cm, '--k')
ax3.plot(tr, R_arr, '--k')
ax1.legend(bbox_to_anchor = (1.1,0))
ax1.set_title(f'Constant Rainfall, dy = L/{m}')
#np.savetxt(f'cr{end}.csv', np.column_stack((h_m, yvals)), delimiter=",", header='h,y')

if finite_diff ==1:
    fig2, (axs1, axs2) = plt.subplots(2)
    for t in profiles:
        fdvals = axs1.plot(yvals, profiles[t], '--')
        axs1.plot(yvals, profiles_el[t], color=fdvals[0].get_color(), label=f"t={int(t)}s")


    axs2.plot(time_array, hcm_hist, '--k')
    axs2.plot(time_array, h_cm, color='red')
    axs1.legend(bbox_to_anchor = (1.1,0))
    axs1.set_ylabel(r'$h_{m}(y,t)$ [m]')
    ax1.set_xlabel(r'$y$ [m]')
    axs2.set_ylabel(r'$h_{cm}(y,t)$ [m]')
    axs2.set_xlabel(r'$t$ [s]')

    axs2.set_xlim(0, 100)
    axs1.set_title(f'dy = Ly / {m}')
plt.show()
plt.figure()


#print(h_m, yvals)
























































































