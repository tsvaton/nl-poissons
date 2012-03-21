#include "hermes2d.h"

using namespace Hermes;
using namespace Hermes::Hermes2D;
using namespace Hermes::Hermes2D::WeakFormsH1;
using namespace Hermes::Hermes2D::Views;

/* Nonlinearity lambda(u) = alpha + Hermes::pow(u, beta) */

class CustomNonlinearity : public Hermes1DFunction<double>
{
public:
  CustomNonlinearity(double alpha, double beta);

  virtual double value(double u) const;

  virtual Ord value(Ord u) const;

  virtual double derivative(double u) const;

  virtual Ord derivative(Ord u) const;

protected:
  double alpha;
  double beta;
};

/* Nonlinearity ksi(u) = delta * Hermes::pow(u,gamma) */

class CustomNonlinearBulk : public Hermes2DFunction<double>
{
public:
  CustomNonlinearBulk(double gamma, double delta);

  virtual double value(double u) const;

  virtual Ord value(Ord u) const;

  virtual double derivative(double u) const;

  virtual Ord derivative(Ord u) const;

protected:
  double gamma;
  double delta;
};


/* Initial condition */
class CustomInitialCondition : public ExactSolutionScalar<double>
{
public:
  CustomInitialCondition(Mesh* mesh) : ExactSolutionScalar<double>(mesh) 
  {
  };

  virtual double value(double x, double y) const;

  virtual void derivatives(double x, double y, double& dx, double& dy) const;

  virtual Ord ord(Ord x, Ord y) const;
};


/* Weak form */
class CustomWeakFormDiffusion : public WeakForm<double>
{
public:
  CustomWeakFormDiffusion(std::string area, Hermes1DFunction<double>* lambda, 
                                            Hermes2DFunction<double>* ksi, 
                                            Hermes2DFunction<double>* src_term);
};



/* Essential boundary conditions */

class CustomEssentialBCNonConst : public EssentialBoundaryCondition<double>
{
public:
  CustomEssentialBCNonConst(std::string marker) 
           : EssentialBoundaryCondition<double>(Hermes::vector<std::string>()) 
  {
    this->markers.push_back(marker);
  }

  virtual EssentialBCValueType get_value_type() const;

  virtual double value(double x, double y, double n_x, double n_y, 
                       double t_x, double t_y) const;
};
