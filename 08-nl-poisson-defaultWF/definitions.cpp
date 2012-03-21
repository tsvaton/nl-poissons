#include "definitions.h"

CustomNonlinearity::CustomNonlinearity(double alpha, double beta): Hermes1DFunction<double>()
{
  this->is_const = false;
  this->alpha = alpha;
  this->beta = beta;
}

double CustomNonlinearity::value(double u) const
{
  return alpha + Hermes::pow(u, beta);
}

Ord CustomNonlinearity::value(Ord u) const
{
  return Ord(10);
}

double CustomNonlinearity::derivative(double u) const
{
  return beta * Hermes::pow(u, beta - 1.0);
}

Ord CustomNonlinearity::derivative(Ord u) const
{
  // Same comment as above applies.
  return Ord(10);
}



CustomNonlinearBulk::CustomNonlinearBulk(double gamma, double delta): Hermes2DFunction<double>()
{
  this->is_const = false;
  this->gamma = gamma;
  this->delta = delta;
}

double CustomNonlinearBulk::value(double u) const
{
  return delta * Hermes::pow(u, gamma);
}

Ord CustomNonlinearBulk::value(Ord u) const
{
  return Ord(10);
}

double CustomNonlinearBulk::derivative(double u) const
{
  return delta * gamma * Hermes::pow(u, gamma - 1.0);
}

Ord CustomNonlinearBulk::derivative(Ord u) const
{
  // Same comment as above applies.
  return Ord(10);
}



CustomWeakFormDiffusion::CustomWeakFormDiffusion(std::string area, Hermes1DFunction<double>* lambda, 
                                                                   Hermes2DFunction<double>* eta, 
                                                                   Hermes2DFunction<double>* src_term) : WeakForm<double>(1)
{
  // Jacobian
  // Contribution of the volumetric term.
  add_matrix_form(new DefaultMatrixFormVol<double>(0, 0, area, eta));
  // Contribution of the diffusion term.
  add_matrix_form(new DefaultJacobianDiffusion<double>(0, 0, area, lambda));

  // Residual
  // Contribution of the volumetric term.
  add_vector_form(new DefaultResidualVol<double>(0, area, eta));
  // Contribution of the diffusion term.
  add_vector_form(new DefaultResidualDiffusion<double>(0, area, lambda));
  // Contribution if the rhs.
  add_vector_form(new DefaultVectorFormVol<double>(0, HERMES_ANY, src_term));
}



/* Initial conditions */
double CustomInitialCondition::value(double x, double y) const 
{
  return (x+10) * (y+10) / 100. + 2;
}

void CustomInitialCondition::derivatives(double x, double y, double& dx, double& dy) const 
{   
  dx = (y+10) / 100.;
  dy = (x+10) / 100.;
}

Ord CustomInitialCondition::ord(Ord x, Ord y) const 
{
  return x*y;
}

EssentialBoundaryCondition<double>::EssentialBCValueType CustomEssentialBCNonConst::get_value_type() const 
{ 
  return EssentialBoundaryCondition<double>::BC_FUNCTION; 
}

double CustomEssentialBCNonConst::value(double x, double y, double n_x, double n_y, 
                                        double t_x, double t_y) const
{
  //return (x+10) * (y+10) / 100.;
  return 1.0;
}
