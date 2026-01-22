#include "SolverBase.h"
#include "problem_instance/TargetFunction.h"

namespace macroplacer
{

SolverBase::SolverBase() : num_var_(0) {}

SolverBase::SolverBase(
  std::shared_ptr<HyperParam>     param, 
  std::shared_ptr<TargetFunction> gp_problem)
{
  param_ = param;
  target_function_ = gp_problem;

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

SolverType
SolverBase::getSolverType() const
{
  return type_;
}

}
