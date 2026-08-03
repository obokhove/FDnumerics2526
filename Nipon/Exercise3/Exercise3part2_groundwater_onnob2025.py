from firedrake import *
# 
# packages Working code as of May 14th 2018 by Will Booker and Onno Bokhove
# import numpy as np
from math import pow
import time as tijd # OB2025
import numpy as np # OB2025
import matplotlib # OB2025
import matplotlib.pyplot as plt  #for dynamic plotting

#Mesh grid layout building with elements 
m  = 32 #for high resolution & 16 for faster calculation time (only to compare pulse durations)
Ly = 0.85
dy = Ly/m
mesh = IntervalMesh(m, 0 , Ly)
# OB2025 y = mesh.coordinates # OB2025 Mesh coordinates
y, = SpatialCoordinate(mesh) # OB2025 

# Time definitions from 0s to 100s
t   = 0.0
end = 100.0 # Running for 100 seconds as requested in Q2
Ntm = 50
dtmeas = 10.0  # Exporting spatial profiles every 10 seconds
tmeas = dtmeas
t_meas_2s = 2.0  # Log scalar data every 2 seconds

#
# Define Function space on our mesh.
# Initially we will use a continuous linear Lagrange basis
# Tried other order, 1, 2
nCG = 2 #Defining Function space (P1 Linear Elements and P2 Quadrilateral)
V = FunctionSpace(mesh, "CG", nCG) # OB2025

# Define timestep value
CFL = 2.3
Dt = CFL*0.5*dy*dy  # Based on FD estimate; note that dt must be defined before flux, etc
# Dt = 16*Dt
# dt.assign(CFL*0.5*dy*dy)

#Dt= 0.1 # for the Crank-Nicholson scheme
dt = Constant(Dt) # Using dt.assign in the while loop should avoid having to rebuild the solver iirc
#theta=0.0 # for Explicit Forward Euler
# Defining Crank Nicholson parameter with theta=0.5
theta = 0.5

# Defining Groundwater constants values given in the assignment
mpor  = 0.3
sigma = 0.8
Lc    = 0.05
kperm = 1e-8
w     = 0.1
R     = 0.000125
nu    = 1.0e-6
g     = 9.81
alpha = kperm/( nu * mpor * sigma )
gam   = Lc/( mpor*sigma )
fac2  = sqrt(g)/( mpor*sigma )
# 
# ncase = 0 Dirichlet bc, ncase = 1 overflow groundwater into canal section with weir equation:
nncase = 1

# Initial condition
# OB2025: h_prev = Function(V)
# OB2025 old stiff commented out: h_prev.interpolate(Expression("0.0"))
# OB2025
h_prev = Function(V).interpolate(0.0 + 0.0*y) # OB2025 IC, I guess hnum = 0.0*y h_prev.interpolate(hnum)

# Create storage for paraview
outfile = VTKFile("./Results/groundwater_onnob.pvd")

# Write IC to file for paraview
outfile.write(h_prev , t = t )

# Define trial and test functions on this function space
# h will be the equivalent to h^n+1 in our timestepping scheme

phi = TestFunction(V)

def flux ( h , phi , R ):  # phi is test function q in (31) and (32)
    return ( alpha * g * h * dot ( grad (h) , grad (phi) ) - (R * phi )/ ( mpor * sigma ) )

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
     F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx
     # Add boundary contributions at y = 0: 
     F2 = ( gam*phi*(h-h_prev)/dt+theta*phi*fac2*np.power(max_value(2.0*h/3.0,0.0),1.5)+(1-theta)*phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1)
     h_problem = NonlinearVariationalProblem( F+F2 , h )
     h_solver = NonlinearVariationalSolver(h_problem, solver_parameters={'mat_type':'aij','ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})

# Arrays for logging data every 2 seconds
time_history = []
h_cm_history = []
rain_history = []

# --- New additions for spatial plotting ---
spatial_profiles_data = {}
spatial_y_nodes = np.linspace(0, Ly, m * nCG + 1) # Correctly getting DoF coordinates for 1D CG space
spatial_profiles_data[t] = h_prev.dat.data_ro[:].copy() # Storing a copy

# Time loop

while (t < end):
    # Periodic Rainfall Logic (4s wet, 6s dry in a 10s cycle)
    cycle_time = t % 10.0
    if cycle_time < 4.0:
        R.assign(R_max)
    else:
        R.assign(0.0)

    # Advancing time step
    t += Dt

    # Solving based on chosen scheme as theta changes
    if theta == 0.0:
        explicit_solver.solve()
        h_prev.assign(out)
    elif theta > 0.0:
        h_solver.solve()
        h_prev.assign(h)

    # Extracting scalar canal height value at node y=0 for logging
    # (Extracting the float data directly from the first mesh node boundary array entry)
    h_cm_val = float(h_prev.dat.data_ro[0])

    # Log scalar values every 2 seconds
    if t >= t_meas_2s:
        time_history.append(t)
        h_cm_history.append(h_cm_val)
        rain_history.append(float(R))
        t_meas_2s += 2.0

    # Exporting spatial data profiles to Paraview file every 10 seconds
    if t > tmeas:
        print(f'Time is: {t:.2f} s | Canal Level: {h_cm_val:.5f} m')
        tmeas = tmeas + dtmeas
        outfile.write(h_prev, t=t)
        spatial_profiles_data[t] = h_prev.dat.data_ro[:].copy() # Store a copy

# Ending while time loop
print("Times:", [round(x,1) for x in time_history[:10]], "...")
print("h_cm :", [round(x,5) for x in h_cm_history[:10]], "...")

# Plotting  Dynamic 1D Time-Series Dual Plot ---
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Plotting the Canal height response curve (Left Axis)
color = 'tab:blue'
ax1.set_xlabel('Time $t$ (seconds)', fontweight='bold')
ax1.set_ylabel('Canal Water Level $h_{cm}$ (m)', color=color, fontweight='bold')
line1 = ax1.plot(time_history, h_cm_history, color=color, linewidth=2, label='Canal Level $h_{cm}$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Instantiating a shared secondary axis to plot the Rainfall square pulses
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Rainfall Intensity $R(t)$ (m/s)', color=color, fontweight='bold')
line2 = ax2.plot(time_history, rain_history, color=color, linewidth=1.5, linestyle='--', label='Rain Pulse $R(t)$')
ax2.tick_params(axis='y', labelcolor=color)

# Combining legends into a single box
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title('Canal Water Height Response to Periodic Rain Pulses ($\\theta =0.5$,$p_2$)', fontweight='bold', pad=15)
plt.tight_layout()
plt.show()


# Plotting Spatial distributions
plt.figure(figsize=(8, 5))

# Looping through and plot selective 10-second snapshots saved during execution
sorted_times = sorted( spatial_profiles_data.keys())
for index, time_snapshot in enumerate(sorted_times):
    #rounding off for 10-second intervals
    if round(time_snapshot) % 10 == 0:
        
        color_gradient = plt.cm.plasma(index / len(sorted_times))
        plt.plot(spatial_y_nodes, spatial_profiles_data[time_snapshot],
                 color=color_gradient, linewidth=1.5, label=f't = {int(round(time_snapshot))} s')

plt.title('Spatial Expansion Profile ($h_m$) Over Time', fontweight='bold', pad=15)
plt.xlabel('Spatial Channel Coordinate $y$ (m)', fontweight='bold')
plt.ylabel('Groundwater Table Height $h_m$ (m)', fontweight='bold')
plt.xlim(0.0, Ly)
plt.ylim(0.0, max([max(profile) for profile in spatial_profiles_data.values()]) * 1.1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Time Snapshots")

plt.tight_layout()
plt.show()

# Preparing data for contour plot
sorted_times_for_contour = sorted(spatial_profiles_data.keys())
spatial_data_matrix = np.array([spatial_profiles_data[t_key] for t_key in sorted_times_for_contour])

# for_contour are 1D arrays

plt.figure(figsize=(10, 6))
contour = plt.contourf(spatial_y_nodes, sorted_times_for_contour, spatial_data_matrix, levels=50, cmap='viridis')
plt.colorbar(contour, label='Groundwater Table Height $h_m$ (m)')

plt.title('Groundwater Table Height Over Space and Time($\\theta =0.5$)', fontweight='bold', pad=15)
plt.xlabel('Spatial Channel Coordinate $y$ (m)', fontweight='bold')
plt.ylabel('Time $t$ (seconds)', fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
