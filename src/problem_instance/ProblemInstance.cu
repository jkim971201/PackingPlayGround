#include "cuda_linalg/CudaVectorAlgebra.h"
#include "ProblemInstance.h"
#include "Painter.h"

namespace macroplacer
{

ProblemInstance::ProblemInstance(std::shared_ptr<Painter> painter) : painter_(painter) {}

int 
ProblemInstance::getNumVariable() const
{
  return num_var_;
}

void
ProblemInstance::exportToSolver(CudaVector<float>& var_from_solver)
{
  // Host to Device
  var_from_solver = h_solution_;
}

}; // namespace macroplacer
