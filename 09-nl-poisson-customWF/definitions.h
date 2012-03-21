#include "hermes2d.h"

using namespace Hermes;
using namespace Hermes::Hermes2D;
using namespace Hermes::Hermes2D::WeakFormsH1;
using namespace Hermes::Hermes2D::Views;

/* Nonlinearity lambda(u) = alpha + Hermes::pow(u, beta) */

class CustomNonlinearStiff : public Hermes1DFunction<double>
{
public:
  CustomNonlinearStiff(double alpha, double beta);

  virtual double value(double u) const;

  virtual Ord value(Ord u) const;

  virtual double derivative(double u) const;

  virtual Ord derivative(Ord u) const;

protected:
  double alpha;
  double beta;
};

/* Nonlinearity ksi(u) = delta * Hermes::pow(u, gamma) */

class CustomNonlinearBulk : public Hermes1DFunction<double>
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



/* Custom Weak form */
class CustomWeakFormAnia : public WeakForm<double>
{
public:
  CustomWeakFormAnia(Hermes1DFunction<double>* lambda, 
                                       Hermes1DFunction<double>* ksi, 
                                       Hermes2DFunction<double>* f);

private:
  class CustomFormMatrixFormVol : public MatrixFormVol<double>
  {
  public:
    CustomFormMatrixFormVol(int i, int j, Hermes1DFunction<double>* lambda, Hermes1DFunction<double>* ksi) : MatrixFormVol<double>(i, j), lambda(lambda), ksi(ksi) {};

    virtual double value(int n, double *wt, Func<double> *u_ext[], Func<double> *u,
                         Func<double> *v, Geom<double> *e, ExtData<double> *ext) const;

    virtual Ord ord(int n, double *wt, Func<Ord> *u_ext[], Func<Ord> *u, Func<Ord> *v,
                    Geom<Ord> *e, ExtData<Ord> *ext) const;

    protected:
      Hermes1DFunction<double>* lambda;
      Hermes1DFunction<double>* ksi;
  };

  class CustomFormVectorFormVol : public VectorFormVol<double>
  {
  public:
    CustomFormVectorFormVol(int i, Hermes1DFunction<double>* lambda, Hermes1DFunction<double>* ksi, Hermes2DFunction<double>* f)
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
  }

  virtual EssentialBCValueType get_value_type() const;

  virtual double value(double x, double y, double n_x, double n_y, 
                       double t_x, double t_y) const;
};
