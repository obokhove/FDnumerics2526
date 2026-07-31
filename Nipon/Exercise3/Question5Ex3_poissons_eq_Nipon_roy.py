#====================================================
# Solution by Firedrake FEM-CG of a Poisson equation
#====================================================
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
from firedrake.pyplot import tripcolor, tricontour

# Mesh grid layout parameters for the domain discretization
nx = ny = 128 # Tried various mesh resolutions, starting coarse, 16x16 to finer 256x256 etc.

# Generating a regular, structured quadrilateral mesh layout
mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
# Quadrilateral regular mesh made: https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh

# Defining linear continuous Galerkin basis functions (Polynomial order, p=1)
V = FunctionSpace(mesh, 'CG', 1) # Piecewise linear continuous Galerkin function space or polynomials
# See: https://www.firedrakeproject.org/variational-problems.html

#
# Method 1: constructing the weak form manually by multiplying and manipulating the Poisson equation and solving the linear system
#
u = TrialFunction(V) # The unknown or variable u(x,y) solution space
v = TestFunction(V)  # The testfunction of u, which may be better called delu or deltau

x, y = SpatialCoordinate(mesh) # Spatial coordinates

f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y)) # The given source function f(x,y) mapping

a = (inner(grad(u),grad(v)))*dx #Step 2/3: The weak form first term
L = (f*v)*dx # Step 2/3: The weak form second term; dx is the infinitesimal piece in the damain here: dx*dy=dA with area A.

u_1 = Function(V, name='u_1') # Name of the solution (u1) for first method
# Dirichlet boundary conditions restricted at x=0 (ID 1) and x=1 (ID 2)
bc_x0 = DirichletBC(V, Constant(0), 1) # Dirichlet boundary conditions imposed for x=0
bc_x1 = DirichletBC(V, Constant(0), 2) # Dirichlet boundary conditions imposed for x=1
# See: https://www.firedrakeproject.org/firedrake.html#firedrake.bcs.DirichletBC
# The homogeneous Neumann boundary conditions are "automatically" set for the top and bottom boundaries.

#Assembling elements into global matrix framework (K u = F) and solving via CG (u1)
#The solution u1 is assigned to u_1
solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1]) 

#
# Method 2: generating the weak form by "derivative()" of the Ritz-Galerkin integral or variational principle 
#
u_2 = Function(V, name='u_2') # Name of solution for 2nd method

#Direct mapping of the continuous Energy Functional J(u)
Ju = (0.5*inner(grad(u_2),grad(u_2)) - u_2*f)*dx # f->ULF? Step 2

F = derivative(Ju, u_2, du=v) # Step 2/3: The weak form generated
#Executing non-linear monitor to minimize global energy
solve(F == 0, u_2, bcs=[bc_x0, bc_x1]) # Step 4: the solution assigned to u2

#
# Post-processing: Using Paraview to visualise
# See https://www.firedrakeproject.org/visualisation.html#creating-output-files
outfile = VTKFile('output.pvd')
outfile.write(u_1, u_2)

f.interpolate(sin(pi*x)*cos(pi*y))
L2_1 = sqrt(assemble(dot(u_1 - f, u_1 - f) * dx)) # L2 error solution u1
L2_2 = sqrt(assemble(dot(u_2 - f, u_2 - f) * dx)) # L2 error solution u2
# difference between the solutions u_1 and u_2
L2 = sqrt(assemble(dot(u_2 - u_1, u_2 - u_1) * dx)) # L2 error difference
print(f'Mesh resolution: Δx = {1/nx}')
print(f'L2 error: Method1 = {L2_1}, Method2 = {L2_2}')
print(f'L2 norm between the two results: {L2}')
#
#Post-processing for plotting the results and calculating the L2 errors
#
# Getting the raw data from the Function u_1
u1_values = u_1.dat.data

# Calculating the min and max values for u_1
min_u1 = np.min(u1_values)
max_u1 = np.max(u1_values)


# Getting the raw data from the Function u_2
u2_values = u_2.dat.data

# Calculating the min and max values for u_2
min_u2 = np.min(u2_values)
max_u2 = np.max(u2_values)


# Creating a single figure with two subplots side by side for plotting the solutions u1 and u2
fig, (axes_u1_2d, axes_u2_2d) = plt.subplots(1, 2, figsize=(16, 6))

# Plotting u_1 on the first subplot
colors_u1_2d = tripcolor(u_1, axes=axes_u1_2d, vmin=min_u1, vmax=max_u1)
cbar_u1_2d = fig.colorbar(colors_u1_2d, ax=axes_u1_2d)
tricontour(u_1, axes=axes_u1_2d, vmin=min_u1, vmax=max_u1)
axes_u1_2d.set_title("Firedrake Solution $u_1$ (2D Visualization)")
axes_u1_2d.set_xlabel("Spatial X-coordinate")
axes_u1_2d.set_ylabel("Spatial Y-coordinate")

# Plotting u_2 on the second subplot
colors_u2_2d = tripcolor(u_2, axes=axes_u2_2d, vmin=min_u2, vmax=max_u2)
cbar_u2_2d = fig.colorbar(colors_u2_2d, ax=axes_u2_2d)
tricontour(u_2, axes=axes_u2_2d, vmin=min_u2, vmax=max_u2)
axes_u2_2d.set_title("Firedrake Solution $u_2$ (2D Visualization)")
axes_u2_2d.set_xlabel("Spatial X-coordinate")
axes_u2_2d.set_ylabel("Spatial Y-coordinate")

plt.tight_layout() # Adjusting the layout to prevent overlapping
plt.show()

#-----------------#
# Interpolating the absolute difference expressions (|u_h-u_e|) into the new Function objects

# Creating Function objects to store the interpolated absolute differences
error_function_u1 = Function(V)
error_function_u2 = Function(V)

# f holds the exact solution after re-interpolation
error_function_u1.interpolate(abs(u_1 - f))
error_function_u2.interpolate(abs(u_2 - f))

# Getting raw data and the min/max for diff_u1 of solution u1
diff_u1_values = error_function_u1.dat.data
min_diff_u1 = np.min(diff_u1_values)
max_diff_u1 = np.max(diff_u1_values)

# Getting the raw data and the min/max for diff_u2 of solution u2
diff_u2_values = error_function_u2.dat.data
min_diff_u2 = np.min(diff_u2_values)
max_diff_u2 = np.max(diff_u2_values)

# Creating a single figure with two subplots side by side for the absolute differences
fig_diff, (axes_diff_u1, axes_diff_u2) = plt.subplots(1, 2, figsize=(16, 6))

# Plotting |u_1 - u_e| on the first subplot
# Using error_function_u1 for plotting 
colors_diff_u1 = tripcolor(error_function_u1, axes=axes_diff_u1, vmin=0, vmax=max(max_diff_u1, max_diff_u2))
cbar_diff_u1 = fig_diff.colorbar(colors_diff_u1, ax=axes_diff_u1)
cbar_diff_u1.set_label("Absolute Error $|u_1 - u_e|$ values")
tricontour(error_function_u1, axes=axes_diff_u1, vmin=0, vmax=max(max_diff_u1, max_diff_u2))
axes_diff_u1.set_title("Absolute Error $|u_1 - u_e|$ (2D Visualization)")
axes_diff_u1.set_xlabel("Spatial X-coordinate")
axes_diff_u1.set_ylabel("Spatial Y-coordinate")

# Plotting |u_2 - u_e| on the second subplot
# Using error_function_u2 for plotting
colors_diff_u2 = tripcolor(error_function_u2, axes=axes_diff_u2, vmin=0, vmax=max(max_diff_u1, max_diff_u2))
cbar_diff_u2 = fig_diff.colorbar(colors_diff_u2, ax=axes_diff_u2)
cbar_diff_u2.set_label("Absolute Error $|u_2 - u_e|$ values")
tricontour(error_function_u2, axes=axes_diff_u2, vmin=0, vmax=max(max_diff_u1, max_diff_u2))
axes_diff_u2.set_title("Absolute Error $|u_2 - u_e|$ (2D Visualization)")
axes_diff_u2.set_xlabel("Spatial X-coordinate")
axes_diff_u2.set_ylabel("Spatial Y-coordinate")

plt.tight_layout() 
plt.show()
#Printing the max error values for u1 and u2   
print(f"Max absolute error for u_1: {max_diff_u1:.4e}")
print(f"Max absolute error for u_2: {max_diff_u2:.4e}")
