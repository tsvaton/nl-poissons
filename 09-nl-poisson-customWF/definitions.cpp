#include "definitions.h"

CustomNonlinearStiff::CustomNonlinearStiff(double alpha, double beta): Hermes1DFunction<double>()
{
  this->is_const = false;
  this->alpha = alpha;
  this->beta = beta;
}

double CustomNonlinearStiff::value(double u) const
{
  return alpha + Hermes::pow(u, beta);
}

Ord CustomNonlinearStiff::value(Ord u) const
{
  return Ord(10);
}

double CustomNonlinearStiff::derivative(double u) const
{
  return beta * Hermes::pow(u, beta - 1.0);
}

Ord CustomNonlinearStiff::derivative(Ord u) const
{
  return Ord(10);
}



CustomNonlinearBulk::CustomNonlinearBulk(double gamma, double delta): Hermes1DFunction<double>()
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
  return gamma * delta * Hermes::pow(u, gamma - 1.0);
}

Ord CustomNonlinearBulk::derivative(Ord u) const
{
  return Ord(10);
}



/* Custom Weak form */
CustomWeakFormAnia::CustomWeakFormAnia(Hermes1DFunction<double>* lambda,
                                       Hermes1DFunction<double>* ksi,
                                       Hermes2DFunction<double>* f) : WeakForm<double>(1)
{
  // Jacobian.
  add_matrix_form(new CustomFormMatrixFormVol(0, 0, lambda, ksi));
  
  // Residual.
  add_vector_form(new CustomFormVectorFormVol(0, lambda, ksi, f));
}


double CustomWeakFormAnia::CustomFormMatrixFormVol::value(int n, double *wt, Func<double> *u_ext[],
                                                   Func<double> *u, Func<double> *v,
                                                   Geom<double> *e, ExtData<double> *ext) const
{
  double result = 0;
  for (int i = 0; i < n; i++)
  {
    result += wt[i] * (lambda->derivative(u_ext[0]->val[i]) * u->val[i] * (u_ext[0]->dx[i] * v->dx[i] + u_ext[0]->dy[i] * v->dy[i])
                     + lambda->value(u_ext[0]->val[i]) * (u->dx[i] * v->dx[i] + u->dy[i] * v->dy[i]));
    result += wt[i] * (ksi->derivative(u_ext[0]->val[i])) * u->val[i] * v->val[i];
  }
  return result;
}

Ord CustomWeakFormAnia::CustomFormMatrixFormVol::ord(int n, double *wt, Func<Ord> *u_ext[],
                                              Func<Ord> *u, Func<Ord> *v,
                                              Geom<Ord> *e, ExtData<Ord> *ext) const
{
  Ord result = Ord(0);
  for (int i = 0; i < n; i++)
  {
    result += wt[i] * (lambda->derivative(u_ext[0]->val[i]) * u->val[i] * (u_ext[0]->dx[i] * v->dx[i] + u_ext[0]->dy[i] * v->dy[i])
                     + lambda->value(u_ext[0]->val[i]) * (u->dx[i] * v->dx[i] + u->dy[i] * v->dy[i]));
    result += wt[i] * (ksi->derivative(u_ext[0]->val[i])) * u->val[i] * v->val[i];
  }
  return result;
}

double CustomWeakFormAnia::CustomFormVectorFormVol::value(int n, double *wt, Func<double> *u_ext[],
                                                   Func<double> *v, Geom<double> *e, ExtData<double> *ext) const
{
  double result = 0;
  for (int i = 0; i < n; i++)
  {
    result += wt[i] * lambda->value(u_ext[0]->val[i])
                    * (u_ext[0]->dx[i] * v->dx[i] + u_ext[0]->dy[i] * v->dy[i]);
    result += wt[i] * ksi->value(u_ext[0]->val[i]) * v->val[i];
    result += wt[i] * f->value(e->x[i], e->y[i]) * v->val[i];
  }
  return result;
}

Ord CustomWeakFormAnia::CustomFormVectorFormVol::ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *v,
                                              Geom<Ord> *e, ExtData<Ord> *ext) const
{
  Ord result = Ord(0);
  for (int i = 0; i < n; i++)
  {
    result += wt[i] * lambda->value(u_ext[0]->val[i])
                    * (u_ext[0]->dx[i] * v->dx[i] + u_ext[0]->dy[i] * v->dy[i]);
    result += wt[i] * ksi->value(u_ext[0]->val[i]) * v->val[i];
    result += wt[i] * f->value(e->x[i], e->y[i]) * v->val[i];
  }
  return result;
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
