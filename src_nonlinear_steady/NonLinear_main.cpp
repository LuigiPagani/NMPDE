#include "NonLinearDiffusion.hpp"
#include <deal.II/base/convergence_table.h>



#ifdef CONVERGENCE
// Main function.
int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  ConvergenceTable table;

  const std::vector<std::string> meshes = {"../mesh/mesh-cube-5.msh",
                                           "../mesh/mesh-cube-10.msh",
                                           "../mesh/mesh-cube-20.msh",
                                           "../mesh/mesh-cube-40.msh"};
  const std::vector<double>      h_vals = {1.0 / 10.0,
                                           1.0 / 20.0,
                                           1.0 / 40.0,
                                           1.0 / 80.0};
  const unsigned int             degree = 1;

  // Only for Exercise 1:
  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (unsigned int i = 0; i < meshes.size(); ++i)
    {
      NonLinearDiffusion problem(meshes[i], degree);

      problem.setup();
      problem.solve_newton();
      problem.output();

      // Only for Exercise 1:
      const double error_L2 = problem.compute_error(VectorTools::L2_norm);
      const double error_H1 = problem.compute_error(VectorTools::H1_norm);

      table.add_value("h", h_vals[i]);
      table.add_value("L2", error_L2);
      table.add_value("H1", error_H1);

      convergence_file << h_vals[i] << "," << error_L2 << "," << error_H1
                       << std::endl;
    }

  // Only for Exercise 1:
  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);

  return 0;
}

#endif //CONVERGENCE

#ifndef CONVERGENCE

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../mesh/mesh-cube-20.msh";
  const unsigned int degree         = 1;

  NonLinearDiffusion problem(mesh_file_name, degree);

  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
#endif //CONVERGENCE