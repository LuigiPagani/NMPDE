#include "NonLinearDiffusion.hpp"
#include <deal.II/base/convergence_table.h>


// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  ConvergenceTable table;


  const std::string  mesh_file_name = "../mesh/mesh-square-h0.012500.msh";
  const unsigned int degree         = 2;

  NonLinearDiffusion problem(mesh_file_name, degree);

  problem.setup();
  problem.solve_newton();
  problem.output();
  const double error_L2 = problem.compute_error(VectorTools::L2_norm);
  const double error_H1 = problem.compute_error(VectorTools::H1_norm);
  printf("L2 error: %f\n", error_L2);
  printf("H1 error: %f\n", error_H1);

  return 0;
}