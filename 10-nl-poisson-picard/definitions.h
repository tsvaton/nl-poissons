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

  protected:
    double alpha;
    double beta;
};


/* Nonlinearity lambda(u) = delta *  Hermes::pow(u, gamma) */

class CustomNonlinearBulk : public Hermes1DFunction<double>
{
public:
  CustomNonlinearBulk(double gamma, double delta);

  virtual double value(double u) const;

  virtual Ord value(Ord u) const;

  protected:
    double gamma;
    double delta;
};



/* Initial condition */

class CustomInitialCondition : public ExactSolutionScalar<double>
{
public:
  CustomInitialCondition(Mesh* mesh, double const_value) : ExactSolutionScalar<double>(mesh), 
    const_value(const_value)
  {
  };

  virtual double value(double x, double y) const;

  virtual void derivatives(double x, double y, double& dx, double& dy) const;

  virtual Ord ord(Ord x, Ord y) const;

  double const_value;
};

/* Weak forms */

// NOTE: The linear problem in the Picard's method is 
//       solved using the Newton's method.

class CustomWeakFormPicard : public WeakForm<double>
{
public:
  CustomWeakFormPicard(Solution<double>* prev_iter_sln, Hermes1DFunction<double>* lambda, 
                                                        Hermes1DFunction<double>* ksi, 
                                                        Hermes2DFunction<double>* f);

private:
  class CustomJacobian : public MatrixFormVol<double>
  {
  public:
    CustomJacobian(int i, int j, Hermes1DFunction<double>* lambda, Hermes1DFunction<double>* ksi) : MatrixFormVol<double>(i, j), lambda(lambda), ksi(ksi) {};

    virtual double value(int n, double *wt, Func<double> *u_ext[], Func<double> *u,
                         Func<double> *v, Geom<double> *e, ExtData<double> *ext) const;

    virtual Ord ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *u, Func<Ord> *v,
                    Geom<Ord> *e, ExtData<Ord> *ext) const;
    
    protected:
      Hermes1DFunction<double>* lambda;
      Hermes1DFunction<double>* ksi;
  };

  class CustomResidual : public VectorFormVol<double>
  {
  public:
    CustomResidual(int i, Hermes1DFunction<double>* lambda, Hermes1DFunction<double>* ksi, Hermes2DFunction<double>* f) 
      : VectorFormVol<double>(i), lambda(lambda), ksi(ksi), f(f) 
    {
    }

    virtual double value(int n, double *wt, Func<double> *u_ext[],
                         Func<double> *v, Geom<double> *e, ExtData<double> *ext) const;

    virtual Ord ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *v, 
                    Geom<Ord> *e, ExtData<Ord> *ext) const;

    private:
      Hermes1DFunction<double>* lambda;
      Hermes1DFunction<double>* ksi;
      Hermes2DFunction<double>* f;
  };
};


/* Essential boundary conditions */
class CustomEssentialBCNonConst : public EssentialBoundaryCondition<double> 
{
public:
  CustomEssentialBCNonConst(std::string marker) 
           : EssentialBoundaryCondition<double>(Hermes::vector<std::string>()) 
  {
    this->markers.push_back(marker);
  };

  virtual EssentialBoundaryCondition<double>::EssentialBCValueType get_value_type() const;

  virtual double value(double x, double y, double n_x, double n_y, 
                       double t_x, double t_y) const;
};
