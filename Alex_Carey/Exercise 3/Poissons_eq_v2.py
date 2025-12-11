#!/usr/bin/env python3
"""
================================================================================
Poisson Equation Solver using Firedrake FEM-CG
================================================================================

SUMMARY:
--------
This script solves the 2D Poisson equation:
    -∇²u = f(x,y)
on the unit square [0,1]×[0,1] with:
    - Homogeneous Dirichlet BCs at x=0 and x=1: u(0,y) = u(1,y) = 0
    - Homogeneous Neumann BCs at y=0 and y=1 (natural BCs, automatically satisfied)
    - Source term: f(x,y) = 2π² sin(πx) cos(πy)
    - Exact solution: u(x,y) = sin(πx) cos(πy)

METHODS:
--------
1. Method 1: Manual weak form construction
   - Constructs bilinear form a(u,v) and linear form L(v) explicitly
   - Solves the resulting linear system directly

2. Method 2: Variational principle via derivative
   - Defines the Ritz-Galerkin functional J[u]
   - Generates weak form F via automatic differentiation
   - Solves the resulting nonlinear system (which is linear for Poisson)

CONTENTS:
---------
1. Mesh generation (quadrilateral elements)
2. Function space definition (CG1 - piecewise linear continuous Galerkin)
3. Method 1: Direct weak form solve
4. Method 2: Variational principle solve
5. Post-processing: VTK output and L2 error computation

OUTPUT:
-------
- output.pvd: ParaView file containing both solutions (u_1, u_2)
- Console: L2 errors for each method and difference between methods

USAGE:
------
Run with: python3 Poissons_eq_v2.py
View with: paraview output.pvd

================================================================================
"""

from firedrake import *

# ==============================================================================
# 1. MESH GENERATION
# ==============================================================================

# Mesh resolution: number of elements in x and y directions
nx = ny = 128  # Try various mesh resolutions, starting coarse (e.g., 16x16)

# Create a structured quadrilateral mesh on the unit square [0,1]×[0,1]
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)
# Reference: https://www.firedrakeproject.org/firedrake.html#firedrake.utility_meshes.UnitSquareMesh
# Alternatively, use gmsh for more complex geometries

# ==============================================================================
# 2. FUNCTION SPACE DEFINITION
# ==============================================================================

# Define piecewise linear continuous Galerkin (CG1) function space
# This means polynomials of degree 1 that are continuous across element boundaries
V = FunctionSpace(mesh, 'CG', 1)
# Reference: https://www.firedrakeproject.org/variational-problems.html

# ==============================================================================
# 3. METHOD 1: MANUAL WEAK FORM CONSTRUCTION
# ==============================================================================
# 
# Weak formulation of -∇²u = f:
#   ∫ ∇u·∇v dx = ∫ f·v dx  for all test functions v
#
# This is obtained by multiplying the PDE by a test function v,
# integrating over the domain, and applying integration by parts.
#

# Define trial function (the unknown u) and test function (v)
u = TrialFunction(V)  # The unknown function u(x,y) to be solved for
v = TestFunction(V)   # The test function v(x,y) (also called δu or variation)

# Get mesh coordinates for defining the source term
x, y = SpatialCoordinate(mesh)

# Define the source term f(x,y) = 2π² sin(πx) cos(πy)
# This is chosen so the exact solution is u(x,y) = sin(πx) cos(πy)
# Since -∇²(sin(πx)cos(πy)) = 2π² sin(πx) cos(πy)
f = Function(V).interpolate(2*pi**2*sin(pi*x)*cos(pi*y))

# Bilinear form: a(u,v) = ∫ ∇u·∇v dx
# This represents the left-hand side of the weak form
a = (inner(grad(u), grad(v)))*dx

# Linear form: L(v) = ∫ f·v dx
# This represents the right-hand side of the weak form
# dx represents the infinitesimal area element dx·dy
L = (f*v)*dx

# Create a Function to hold the solution from Method 1
u_1 = Function(V, name='u_1')

# Define Dirichlet boundary conditions
# bc_x0: u = 0 on the boundary at x=0 (boundary ID 1)
bc_x0 = DirichletBC(V, Constant(0), 1)
# bc_x1: u = 0 on the boundary at x=1 (boundary ID 2)
bc_x1 = DirichletBC(V, Constant(0), 2)
# Reference: https://www.firedrakeproject.org/firedrake.html#firedrake.bcs.DirichletBC
# Note: Homogeneous Neumann BCs (∂u/∂n = 0 at y=0,1) are natural boundary conditions
#       and are automatically satisfied in the weak form (no explicit implementation needed)

# Solve the linear system a(u,v) = L(v) subject to boundary conditions
# Solver parameters:
#   - ksp_type='cg': Use Conjugate Gradient iterative solver
#   - pc_type='none': No preconditioner (for demonstration; in practice use preconditioners)
solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0, bc_x1])

# ==============================================================================
# 4. METHOD 2: VARIATIONAL PRINCIPLE VIA DERIVATIVE
# ==============================================================================
#
# Alternatively, we can define the Ritz-Galerkin functional (energy functional):
#   J[u] = ∫ (½|∇u|² - fu) dx
#
# The solution minimizes this functional. Taking the first variation (derivative)
# with respect to u gives the weak form automatically.
#

# Create a Function to hold the solution from Method 2
u_2 = Function(V, name='u_2')

# Define the Ritz-Galerkin functional J[u]
# J[u] = ∫ (½ ∇u·∇u - u·f) dx
# The minimizer of J[u] satisfies -∇²u = f
Ju = (0.5*inner(grad(u_2), grad(u_2)) - u_2*f)*dx

# Compute the first variation (Gateaux derivative) of J with respect to u_2
# in the direction of the test function v
# This automatically generates the weak form: F(u,v) = 0
F = derivative(Ju, u_2, du=v)

# Solve the nonlinear system F(u,v) = 0 subject to boundary conditions
# (For the Poisson equation, this is actually linear, but the solver handles it as nonlinear)
solve(F == 0, u_2, bcs=[bc_x0, bc_x1])

# ==============================================================================
# 5. POST-PROCESSING
# ==============================================================================

# Write solutions to ParaView-compatible VTK file for visualization
# Reference: https://www.firedrakeproject.org/visualisation.html#creating-output-files
outfile = VTKFile('output.pvd')
outfile.write(u_1, u_2)

# Compute L2 errors by comparing with the exact solution
# Exact solution: u_exact(x,y) = sin(πx) cos(πy)
f.interpolate(sin(pi*x)*cos(pi*y))  # Reuse f to store the exact solution

# L2 error for Method 1: ||u_1 - u_exact||_L2
# L2 norm is: ||w||_L2 = sqrt(∫ w² dx)
L2_1 = sqrt(assemble(dot(u_1 - f, u_1 - f) * dx))

# L2 error for Method 2: ||u_2 - u_exact||_L2
L2_2 = sqrt(assemble(dot(u_2 - f, u_2 - f) * dx))

# L2 norm of difference between the two methods: ||u_2 - u_1||_L2
# This should be near machine precision since both methods solve the same problem
L2 = sqrt(assemble(dot(u_2 - u_1, u_2 - u_1) * dx))

# Print results to console
print(f'Mesh resolution: Δx = {1/nx}')
print(f'L2 error: Method1 = {L2_1}, Method2 = {L2_2}')
print(f'L2 norm between the two results: {L2}')

# ==============================================================================
# END OF SCRIPT
# ==============================================================================