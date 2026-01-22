#ifndef SOLVER_BASE_H 
#define SOLVER_BASE_H

#include <memory>

#include "cuda_linalg/CudaVector.h"

namespace macroplacer
{

using namespace cuda_linalg;

class TargetFunction;

class SolverBase 
{
  public:

    SolverBase();
    SolverBase(std::shared_ptr<TargetFunction> problem);

    // Pure virtual function
    virtual void solve() = 0;

  protected:

    // Pure virtual function
    virtual void initSolver() = 0;
    virtual void initForCUDAKernel() = 0;
    virtual void setInitialSolution() = 0;
    virtual void updateOneIteration(int iter) = 0;

    void initializeSolverBase(std::shared_ptr<TargetFunction> target_function);

    int num_var_;

    std::shared_ptr<TargetFunction> target_function_;

    /* Device Array */
    CudaVector<float> d_curX_;
    CudaVector<float> d_curY_;
    CudaVector<float> d_nextX_;
    CudaVector<float> d_nextY_;
};

} 

#endif
