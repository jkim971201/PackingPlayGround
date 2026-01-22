#include "SolverBase.h"
#include "TargetFunction.h"

namespace macroplacer
{

SolverBase::SolverBase() : num_var_(0) {}

SolverBase::SolverBase(std::shared_ptr<TargetFunction> problem)
{
  target_function_ = problem;
  initializeSolverBase(target_function_);
}

void
SolverBase::initializeSolverBase(std::shared_ptr<TargetFunction> target_function)
{
  num_var_ = target_function->getNumVariable();

  d_curX_.resize(num_var_);
  d_curY_.resize(num_var_);

  d_nextX_.resize(num_var_);
  d_nextY_.resize(num_var_);
}

}
