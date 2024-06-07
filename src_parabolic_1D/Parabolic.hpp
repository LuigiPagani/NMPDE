#ifndef PARABOLIC_HPP
#define PARABOLIC_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

#define NEUMANN
#define ROBIN
#define CONVERGENCE
//#define SPATIAL_CONVERGENCE
#define TRANSPORT_COEFFICIENT
#define MUCOEFFICIENT
#define REACTION_COEFFICIENT
#define CONSERVATIVE_TRANSPORT_COEFFICIENT

// Class representing the non-linear diffusion problem.
class Parabolic
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

#ifdef MUCOEFFICIENT
  // Function for the mu coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };
#endif //MUCOEFFICIENT

#ifdef TRANSPORT_COEFFICIENT
  class TransportCoefficient : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      values[0] = 1.0;
      //values[1] = 1.0;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return 1.0;
      else
        return 1.0;
    }

  };
#endif //TRANSPORT_COEFFICIENT


#ifdef CONSERVATIVE_TRANSPORT_COEFFICIENT
   class ConsTransportCoefficient : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      values[0] = 1.0;
      //values[1] = 1.0;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return 1.0;
      else
        return 1.0;
    }

  };
#endif //CONSERVATIVE_TRANSPORT_COEFFICIENT




#ifdef REACTION_COEFFICIENT
  // Reaction coefficient.
  class ReactionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    ReactionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };
#endif //REACTION_COEFFICIENT


#ifdef ROBIN
  class FunctionGamma : public Function<dim>
  {
  public:
    // Constructor.
    FunctionGamma()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;

    }
  };

  #endif //ROBIN

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {   
      double t = get_time();
      return 2.0* std::exp(-t);
    }
  };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return p[0];
    }
  };

  // Function for the initial condition.
  class FunctionG : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double t = get_time();
      return 0.0;
    }
  };

#ifdef NEUMANN
   // Neumann boundary conditions.
  class FunctionH : public Function<dim>
  {
  public:
    // Constructor.
    FunctionH()
    {}

    // Evaluation:
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      double t = get_time();
      return std::exp(-t);
    }
  };
#endif //NEUMANN

    // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double t = get_time();
      return std::exp(-t)*p[0];
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      // duex / dx
      result[0] =0.0;

      // duex / dy
      result[1] = 0.0;


      // duex / dz
      //result[2] = 0;

      return result;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Parabolic(const std::string  &mesh_file_name_,
       const unsigned int &r_,
       const double       &T_,
       const double       &deltat_,
       const double       &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type);

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices(const double &time);

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void
  solve_time_step();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

#ifdef MUCOEFFICIENT
  // mu coefficient.
  FunctionMu mu;
#endif //MUCOEFFICIENT


#ifdef TRANSPORT_COEFFICIENT
  TransportCoefficient transport_coefficient;
#endif //TRANSPORT_COEFFICIENT

#ifdef CONSERVATIVE_TRANSPORT_COEFFICIENT
  ConsTransportCoefficient cons_transport_coefficient;
#endif //CONSERVATIVE_TRANSPORT_COEFFICIENT

#ifdef REACTION_COEFFICIENT
  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;
#endif //REACTION_COEFFICIENT

#ifdef ROBIN
  FunctionGamma function_gamma;
  
#endif //ROBIN

  // Forcing term.
  ForcingTerm forcing_term;
  
  // Initial condition.
  FunctionU0 u_0;

  // g(x).
  FunctionG function_g;

  // Exact solution.
  ExactSolution exact_solution;


  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

#ifdef NEUMANN
  // Quadrature formula used on boundary lines.
  // h(x).
  FunctionH function_h;
#endif //NEUMANN

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif