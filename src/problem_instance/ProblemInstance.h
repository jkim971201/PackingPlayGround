#ifndef PROBLEM_INSTANCE_H
#define PROBLEM_INSTANCE_H

#include <limits>
#include "cuda_linalg/CudaVector.h"

constexpr float k_float_max = std::numeric_limits<float>::infinity();

namespace macroplacer
{

class Macro;
class Net;
class Pin;

class Painter;

using namespace cuda_linalg;

class ProblemInstance
{
  public:

    // Constructor
    ProblemInstance(std::shared_ptr<Painter> painter);

    // APIs
    // var : {x_vector, y_vector, ... }
    virtual void updatePointAndGetGrad(
      const CudaVector<float>& var,
            CudaVector<float>& grad) = 0;

    virtual void getInitialGrad(
      const CudaVector<float>& initial_var,
            CudaVector<float>& initial_grad) = 0;

    virtual void clipToFeasibleSolution(CudaVector<float>& var) = 0;

    virtual void updateParameters() = 0;

    virtual void solveBgnCbk() = 0;
    virtual void solveEndCbk(int iter, double runtime, const CudaVector<float>& var) = 0;

    virtual void iterBgnCbk(int iter) = 0;
    virtual void iterEndCbk(int iter, double runtime, const CudaVector<float>& var) = 0;

    virtual bool checkConvergence() const = 0;

    // Non Virtual functions
    void exportToSolver(CudaVector<float>& var_from_solver); 
    // This will be used only to get initial solution

    int getNumVariable() const;

  protected:

    int num_var_;

    bool plot_mode_;

    std::vector<float> h_solution_;

    std::shared_ptr<Painter> painter_;
};

}; // namespace macroplacer

#endif
