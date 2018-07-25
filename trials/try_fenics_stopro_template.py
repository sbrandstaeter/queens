#!/usr/bin/python3
"""
Poisson equation with Dirichlet conditions and heterogeneous parameter field

"""

# import
import argparse
import os
import fenics as fen
import numpy as np
import sympy as sym

def get_options():
    """ Parse options from command line """

    parser = argparse.ArgumentParser(description="FENICS")
    parser.add_argument('--output_file', type=str, default='fenics.out',
                        help='Outputfile for results')
    args = parser.parse_args()

    if args.output_file is None:
        raise Exception("No output file was given.")

    output_file = os.path.realpath(os.path.expanduser(args.output_file))

    options = {}
    options["output_file"] = output_file

    return  options

def main():
    """ Run FENICS """
    ############################################################################
    # Input parameters
    ############################################################################

    # get options from command line parser
    options = get_options()

    coeffs = [1, 1, 1, 1, 1, 1]
    level = 3
    M0 = 4
    mesh_param = int(M0 * 2**level)
    mat_param = {test_kappa}
    mat_param2 = {test_stopro}
    kappa = fen.Expression('mat_param', degree=0, mat_param=mat_param)

    ############################################################################
    # FEM computation
    ############################################################################

    # Create mesh and define function space
    mesh = fen.UnitSquareMesh(mesh_param, mesh_param)
    V = fen.FunctionSpace(mesh, 'P', 1)

    ############################################################################
    # Define potential field (symbolically) and derive flux boundary condition
    # as well as a compatible dirichlet boundary condition
    ############################################################################
    x0, x1 = sym.symbols('x[0], x[1]')
    coeffs = [1, 1, 1, 1, 1, 1]
    T_kappa = (coeffs[0] + coeffs[1]*x0 + coeffs[2]*x1 + coeffs[3]*x0**2 + \
               coeffs[4]*x1**2 + coeffs[5]*x0*x1) * mat_param

    T = coeffs[0] + coeffs[1]*x0 + coeffs[2]*x1 + coeffs[3]*x0**2 + \
        coeffs[4]*x1**2 + coeffs[5]*x0*x1

    gradx_sym = sym.diff(T_kappa, x0)  # compute g(x0 = 1) = -dT/dn * kappa
    grady_sym = sym.diff(T_kappa, x1)  # compute g(x1 = 1) = -dT/dn * kappa

    rhs = -2*(coeffs[3] + coeffs[4])*mat_param

    gx = fen.Expression(sym.printing.ccode(gradx_sym), degree=1)
    gy = fen.Expression(sym.printing.ccode(grady_sym), degree=1)
    u_D = fen.Expression(sym.printing.ccode(T), degree=2)


    ############################################################################
    # Define subdomains on the boundary, set respective markers and redefine
    # boundary measure ds. The subdomains are defined such that the dirichlet bc
    # can be set to act only on the outer surface of the single element in the
    # lower left corner.
    ############################################################################

    # no idea why 1/meshparam won't work, so apparently this is necessary
    edgesize = np.exp(np.log(1)-np.log(mesh_param))

    class Boundary_D1(fen.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and fen.near(x[1], 0, tol) and (x[0] < edgesize + tol)

    class Boundary_D2(fen.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and fen.near(x[0], 0, tol) and x[1] < edgesize + tol

    class Boundary_N1(fen.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and (fen.near(x[1], 0, tol) and x[0] > edgesize - tol)

    class Boundary_N2(fen.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and fen.near(x[0], 1, tol)

    class Boundary_N3(fen.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and fen.near(x[1], 1, tol)

    class Boundary_N4(fen.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and (fen.near(x[0], 0, tol) and x[1] > edgesize - tol)

    # Mark boundaries
    boundary_markers = fen.FacetFunction('size_t', mesh)
    boundary_markers.set_all(9999)
    b_N1 = Boundary_N1()
    b_N2 = Boundary_N2()
    b_N3 = Boundary_N3()
    b_N4 = Boundary_N4()
    b_D1 = Boundary_D1()
    b_D2 = Boundary_D2()
    b_N1.mark(boundary_markers, 1)
    b_N2.mark(boundary_markers, 2)
    b_N3.mark(boundary_markers, 3)
    b_N4.mark(boundary_markers, 4)
    b_D1.mark(boundary_markers, 0)
    b_D2.mark(boundary_markers, 5)

    # Redefine boundary integration measure
    ds = fen.Measure('ds', domain=mesh, subdomain_data=boundary_markers)


    ############################################################################
    # Compose and solve the FEM-system
    ############################################################################

    # Collect Dirichlet conditions
    bcs = [None]*2
    bcs[0] = fen.DirichletBC(V, u_D, boundary_markers, 0)
    bcs[1] = fen.DirichletBC(V, u_D, boundary_markers, 5)

    # Define variational problem
    u = fen.TrialFunction(V)
    v = fen.TestFunction(V)
    f = fen.Constant(rhs)
    a = kappa*fen.dot(fen.grad(u), fen.grad(v))*fen.dx
    L = f*v*fen.dx - gy*v*ds(1) + gx*v*ds(2) + gy*v*ds(3) - gx*v*ds(4)

    # Compute solution
    u = fen.Function(V)
    fen.solve(a == L, u, bcs)


    ############################################################################
    # save parameter field to file in VTK format
    ############################################################################

    # Save solution to file in VTK format
    vtkfile = fen.File('poisson/solution.pvd')
    vtkfile << u

    ############################################################################
    # Post process the solution (e.g. compute error and flux)
    ############################################################################

    # Compute error in L2 norm
    error_L2 = fen.errornorm(u_D, u, 'L2')

    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

    # Print errors
    print('error_L2  =', error_L2)
    print('error_max =', error_max)


    # computed flux over right facet
    n = fen.FacetNormal(mesh)
    flux = -fen.dot(fen.grad(u), n)*ds(2)
    total_flux = fen.assemble(flux)

    print('Computed flux over right facet: ' + str(total_flux))
    theo_flux = -coeffs[1]-2*coeffs[3]-0.5*coeffs[5]
    print('Exact flux over right facet: ' + str(theo_flux))

    file_object = open(options["output_file"], 'w')
    file_object.write(str(total_flux))
    file_object.close()

if __name__ == '__main__':
    main()
