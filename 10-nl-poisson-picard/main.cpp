#define HERMES_REPORT_ALL
#define HERMES_REPORT_FILE "application.log"
#include "definitions.h"
#include "function/function.h"

using namespace RefinementSelectors;

//  This example uses the Picard's method to solve a nonlinear problem.
//  Try to run this example with PICARD_NUM_LAST_ITER_USED = 1 for 
//  comparison (Anderson acceleration turned off).)
//
//  PDE: -div[lambda(u) grad u] + ksi(u) + src(x, y) = 0.
//
//  Nonlinearity: lambda(u) = alpha + Hermes::pow(u, beta).
//  Nonlinearity: ksi(u) = delta * Hermes::pow(u, gama).
//
//  Picard's linearization: -div[lambda(u^n) grad u^{n+1}] + delta*(u^n)^{gama-1}*u^{n+1} + src(x, y) = 0.
//
//  Domain: square (0, 1)^2.
//
//  BC: 1 Dirichlet on the whole boundary.
//
//  The following parameters can be changed:

// Initial polynomial degree.
const int P_INIT = 2;                             
// Number of initial uniform mesh refinements.
const int INIT_GLOB_REF_NUM = 3;                  
// Number of initial refinements towards boundary.
const int INIT_BDY_REF_NUM = 5;                   
// Value for custom constant initial condition.
const double INIT_COND_CONST = 3.0;               
// Matrix solver: SOLVER_AMESOS, SOLVER_AZTECOO, SOLVER_MUMPS,
// SOLVER_PETSC, SOLVER_SUPERLU, SOLVER_UMFPACK.
MatrixSolverType matrix_solver = SOLVER_UMFPACK;  

// Picard's method.
// Number of last iterations used. 
// 1... standard fixed point.
// >1... Anderson acceleration.
const int PICARD_NUM_LAST_ITER_USED = 4;          
// 0 <= beta <= 1... parameter for the Anderson acceleration. 
const double PICARD_ANDERSON_BETA = 0.2;          
// Stopping criterion for the Picard's method.
const double PICARD_TOL = 1e-3; 
// Maximum allowed number of Picard iterations.
const int PICARD_MAX_ITER = 100;                  

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

  // Initialize previous iteration solution for the Picard's method.
  CustomInitialCondition sln_prev_iter(&mesh, INIT_COND_CONST);

  // Initialize the weak formulation.
  CustomNonlinearity lambda(alpha,beta);
  CustomNonlinearBulk ksi(gama,delta);
  Hermes2DFunction<double> src(-heat_src);
  CustomWeakFormPicard wf(&sln_prev_iter, &lambda, &ksi, &src);

  // Initialize the FE problem.
  DiscreteProblem<double> dp(&wf, &space);

  // Initialize the Picard solver.
  PicardSolver<double> picard(&dp, &sln_prev_iter, matrix_solver);

  // Perform the Picard's iteration (Anderson acceleration on by default).
  if (!picard.solve(PICARD_TOL, PICARD_MAX_ITER, PICARD_NUM_LAST_ITER_USED, 
                    PICARD_ANDERSON_BETA)) error("Picard's iteration failed.");

  // Translate the coefficient vector into a Solution. 
  Solution<double> sln;
  Solution<double>::vector_to_solution(picard.get_sln_vector(), &space, &sln);
  
  // Visualise the solution and mesh.
  ScalarView s_view("Solution", new WinGeom(0, 0, 440, 350));
  s_view.show_mesh(false);
  s_view.show(&sln_prev_iter);
  OrderView o_view("Mesh", new WinGeom(450, 0, 420, 350));
  o_view.show(&space);

  // Wait for all views to be closed.
  View::wait();
  return 0;
}

