#
# Solution by Firedrake FEM-CG of a Poisson equatiom
#
from firedrake import *
import matplotlib.pyplot as plt
from scipy.special import erf


nx = ny = 128 # Try various mesh resolutions, starting coarse, say 16x16 etc.

#f_choice chooses function and bcs
# 0 is original poisson equation
# 1 is f = 5 with original BCs
# 2 is f = 5 with u(0,y)=0, u(1,y) =1
# 3 is f =-6(1-2y)x with u(0,y) = 0, u(1,y) = y^2(3-2y)
f_choice = 3

mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
xcoord = np.linspace(0,1,nx+1)
ycoord = np.linspace(0,1,ny+1)
xc, yc = np.meshgrid(xcoord, ycoord)

# Quadrilateral regular mesh made: https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh
# Alternatively use gmsh:

#creates discrete function space (step 2)
V = FunctionSpace(mesh, 'CG', 1) # Piecewise linear continuous Galerkin function space or polynomials
# See: https://www.firedrakeproject.org/variational-problems.html

#
# Method 1: construct the weak form manually by multiplying and manipulating the Poisson equation and solve the linear system
# This method is the same as STEP 1 creating function and test function
u = TrialFunction(V) # The unknown or variable u(x,y)
v = TestFunction(V)  # The testfunction of u, which may be better called delu or deltau

x, y = SpatialCoordinate(mesh) # Mesh coordinates

if f_choice == 0:
    #original function
    f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y)) # The given function f(x,y)
    uexact = np.sin(np.pi * xc) * np.cos(np.pi * yc)
    bc_x0 = DirichletBC(V, Constant(0), 1)  # Dirichlet boundary conditions imposed
    bc_x1 = DirichletBC(V, Constant(0), 2)  # Dirichlet boundary conditions imposed
elif f_choice == 1:
    #constant function
    f = Function(V).interpolate(5)
    uexact = 5/2 * (xc - xc**2)
    bc_x0 = DirichletBC(V, Constant(0), 1)  # Dirichlet boundary conditions imposed
    bc_x1 = DirichletBC(V, Constant(0), 2)  # Dirichlet boundary conditions imposed
elif f_choice == 2:
    #constant function with u(0,y)=0, u(1,y) =1
    f = Function(V).interpolate(5)
    uexact = 1/2 * (7*xc - 5*xc**2)
    bc_x0 = DirichletBC(V, Constant(0), 1)  # Dirichlet boundary conditions imposed
    bc_x1 = DirichletBC(V, Constant(1), 2)  # Dirichlet boundary conditions imposed
elif f_choice ==3:
    # f =-6(1-2y)x
    # u(0,y) = 0, u(1,y) = y^2(3-2y)
    f = Function(V).interpolate(-6*(1-2*y)*x)
    uexact = yc**2 *(3-2*yc)*xc
    bc_x0 = DirichletBC(V, Constant(0), 1)  # Dirichlet boundary conditions imposed
    bc_x1 = DirichletBC(V, y**2 *(3-2*y), 2)  # Dirichlet boundary conditions imposed
#trying different functions
# Gaussian function
# constant function

#Step 1 creating weak form using function and test function
a = (inner(grad(u),grad(v)))*dx # Step 2/3: The weak form first term
L = (f*v)*dx # Step 2/3: The weak form second term; dx is the infinitesimal piece in the damain here: dx*dy=dA with area A.
# Weak form generated

#discrete u function (step 2)
u_1 = Function(V, name='u_1') # Name of solution for first method



#bc_x0 = DirichletBC(V, Constant(0), 1) # Dirichlet boundary conditions imposed
#bc_x1 = DirichletBC(V, Constant(0), 2) # Dirichlet boundary conditions imposed
# See: https://www.firedrakeproject.org/firedrake.html#firedrake.bcs.DirichletBC
# The homogeneous Neumann boundary conditions are "automatically" included, i.e. do not need anything explicit

#If it wasn't automated, this is where step 3 would be
# and local, reference coordinate system would be introduced

#Firedrake Implementation Step 4
solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1]) # Step 4: the solution assigned to u1

#
# Method 2: generate the weak form via "derivative()" of the Ritz-Galerkin integral or variational principle and solve the nonlinear system
u_2 = Function(V, name='u_2') # Name of solution for first method

Ju = (0.5*inner(grad(u_2),grad(u_2)) - u_2*f)*dx #This is integrand of Ritz-Galerkin integral

F = derivative(Ju, u_2, du=v) # The weak form generated
#Taking derivative of integrand where v is the test function.
#This is step 1 where weak form is derived
#Thjis would be where local and reference coordinates are introduced
# As the system is then solved
solve(F == 0, u_2, bcs=[bc_x0, bc_x1]) # F=0 for all u, so solves this

#
# Post-processing: Use Paraview to visualise
# See https://www.firedrakeproject.org/visualisation.html#creating-output-files
#outfile = VTKFile('output.pvd')
#outfile.write(u_1)
#outfile.write(u_2)




#looping over to set up uvals properly
z1 = np.zeros((nx+1, ny+1)) #vals for u1
z2 = np.zeros((nx+1, ny+1)) #vals for u2
for i in range(nx+1):
    for j in range(ny+1):
        z1[i, j] = u_1.at([xc[i,j], yc[i,j]])
        z2[i, j] = u_2.at([xc[i,j], yc[i,j]])

plt.figure(1)
contour1 = plt.contourf(xc, yc, z1, levels=50)
plt.colorbar(contour1)
plt.title('Numerical Solution')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()
plt.figure(2)
contour2 = plt.contourf(xc, yc, uexact, levels=50)
plt.colorbar(contour2)
plt.title('Exact Solution')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()
plt.figure(3)
udiff = np.abs(z1-uexact)
contour3 = plt.contourf(xc, yc, udiff, levels=50)
plt.colorbar(contour3)
plt.title('|u_h - u_e|')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

f.interpolate(sin(pi*x)*cos(pi*y))
L2_1 = sqrt(assemble(dot(u_1 - f, u_1 - f) * dx)) # L2 error solution u1
L2_2 = sqrt(assemble(dot(u_2 - f, u_2 - f) * dx)) # L2 error solution u2
L2 = sqrt(assemble(dot(u_2 - u_1, u_2 - u_1) * dx)) # L2 error difference
print(f'Mesh resolution: Δx = {1/nx}')
print(f'L2 error: Method1 = {L2_1}, Method2 = {L2_2}')
print(f'L2 norm between the two results: {L2}')