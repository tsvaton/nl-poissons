#define HERMES_REPORT_ALL
#define HERMES_REPORT_FILE "application.log"
#include "definitions.h"
#include "function/function.h"

using namespace RefinementSelectors;

//  Newton's method.
//
//  PDE: Stationary heat transfer equation with nonlinear thermal and source
//       conductivity, - div[lambda(u) grad u] + ksi(u) + src(x,y) = 0.
//
//  Nonlinearity: lambda(u) = alpha + Hermes::pow(u, beta).
//  Nonlinearity: ksi(u) = delta * Hermes::pow(u, gamma).
//  alpha, beta, gamma, delta are real numbers.
//
//  Domain: square (0, 1)^2.
//
//  BC: Constant Dirichlet = 1.
//
//  The following parameters can be changed:

// Initial polynomial degree.
const int P_INIT = 2;                             
// Stopping criterion for the Newton's method.
const double NEWTON_TOL = 1e-8;                   
// Maximum allowed number of Newton iterations.
const int NEWTON_MAX_ITER = 100;                  
// Number of initial uniform mesh refinements.
const int INIT_GLOB_REF_NUM = 3;                  
// Number of initial refinements towards boundary.
const int INIT_BDY_REF_NUM = 4;                   
// Matrix solver: SOLVER_AMESOS, SOLVER_AZTECOO, SOLVER_MUMPS,
// SOLVER_PETSC, SOLVER_SUPERLU, SOLVER_UMFPACK.
MatrixSolverType matrix_solver = SOLVER_UMFPACK;  

// Problem parameters.
double heat_src = 50.0;
double alpha = 0.0;
double beta = 20.0;
double gama = 1.0;
double delta = 2.0;

int main(int argc, char* argv[])
{
  // Load the mesh.
  Mesh mesh;
  MeshReaderH2D mloader;
  mloader.load("square.mesh", &mesh);

  // Perform initial mesh refinements.
  for(int i = 0; i < INIT_GLOB_REF_NUM; i++) mesh.refine_all_elements();
  mesh.refine_towards_boundary("Bdy", INIT_BDY_REF_NUM);

  // Initialize boundary conditions.
  CustomEssentialBCNonConst bc_essential("Bdy");
  EssentialBCs<double> bcs(&bc_essential);

  // Create an H1 space with default shapeset.
  H1Space<double> space(&mesh, &bcs, P_INIT);
  int ndof = space.get_num_dofs();
  info("ndof: %d", ndof);

  // Initialize the weak formulation
  CustomNonlinearStiff lambda(alpha, beta);
  CustomNonlinearBulk ksi(gama, delta);
  Hermes2DFunction<double> src(-heat_src);
  CustomWeakFormAnia wf(&lambda, &ksi, &src);

  // Initialize the FE problem.
  DiscreteProblem<double> dp(&wf, &space);

  // Project the initial condition on the FE space to obtain initial 
  // coefficient vector for the Newton's method.
  // NOTE: If you want to start from the zero vector, just define 
  // coeff_vec to be a vector of ndof zeros (no projection is needed).
  info("Projecting to obtain initial vector for the Newton's method.");
  double* coeff_vec = new double[ndof];
  CustomInitialCondition init_sln(&mesh);
  OGProjection<double>::project_global(&space, &init_sln, coeff_vec, matrix_solver); 

  // Initialize Newton solver.
  NewtonSolver<double> newton(&dp, matrix_solver);

  // Perform Newton's iteration.
  try
  {
    newton.solve(coeff_vec, NEWTON_TOL, NEWTON_MAX_ITER);
  }
  catch(Hermes::Exceptions::Exception e)
  {
    e.printMsg();
    error("Newton's iteration failed.");
  }

  // Translate the resulting coefficient vector into a Solution.
  Solution<double> sln;
  Solution<double>::vector_to_solution(newton.get_sln_vector(), &space, &sln);

  // Get info about time spent during assembling in its respective parts.
  //dp.get_all_profiling_output(std::cout);

  // Clean up.
  delete [] coeff_vec;

  // Visualise the solution and mesh.
  ScalarView s_view("Solution", new WinGeom(0, 0, 440, 350));
  s_view.show_mesh(false);
  s_view.show(&sln);
  OrderView o_view("Mesh", new WinGeom(450, 0, 400, 350));
  o_view.show(&space);

  // Wait for all views to be closed.
  View::wait();
  return 0;
}

