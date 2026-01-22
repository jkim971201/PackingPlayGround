#ifndef ADAM_SOLVER_H
#define ADAM_SOLVER_H

#include "SolverBase.h"

namespace macroplacer
{

class AdamSolver : public SolverBase
{
  public:

    AdamSolver();
    AdamSolver(
      std::shared_ptr<HyperParam> param, 
      std::shared_ptr<TargetFunction> gp_problem);

    // Main Loop
    void solve() override;

  private:

    void initSolver() override;
    void initForCUDAKernel() override;
    void setInitialSolution() override;
    void updateOneIteration(int iter) override;

    void moveForward(
      const float stepLength,
      const CudaVector<float>& d_curX,
      const CudaVector<float>& d_curY,
      const CudaVector<float>& d_curDirectionX,
      const CudaVector<float>& d_curDirectionY,
            CudaVector<float>& d_nextX,
            CudaVector<float>& d_nextY);

    CudaVector<float> d_curDirectionX_;
    CudaVector<float> d_curDirectionY_;

    CudaVector<float> d_curGradX_;
    CudaVector<float> d_curGradY_;

    // Adam Optimizer
    float alpha_;
    float beta1_;
    float beta2_;

    float epsilon_;

    float beta1k_; // beta1_ ^ (k+1)
    float beta2k_; // beta2_ ^ (k+1)

    // First Moment
    CudaVector<float> d_curMX_;
    CudaVector<float> d_curMY_;

    CudaVector<float> d_nextMX_;
    CudaVector<float> d_nextMY_;

    // Second Moment 
    CudaVector<float> d_curNX_;
    CudaVector<float> d_curNY_;

    CudaVector<float> d_nextNX_;
    CudaVector<float> d_nextNY_;

    // Bias Corrected
    CudaVector<float> d_bcMX_;
    CudaVector<float> d_bcMY_;

    CudaVector<float> d_bcNX_;
    CudaVector<float> d_bcNY_;

    void updateMoment(
      const CudaVector<float>& d_curMX,
      const CudaVector<float>& d_curMY,
      const CudaVector<float>& d_curNX,
      const CudaVector<float>& d_curNY,
      const CudaVector<float>& d_curGradX,
      const CudaVector<float>& d_curGradY,
            CudaVector<float>& d_nextMX,
            CudaVector<float>& d_nextMY,
            CudaVector<float>& d_nextNX,
            CudaVector<float>& d_nextNY);

    void updateDirection(
      const CudaVector<float>& d_nextMX,
      const CudaVector<float>& d_nextMY,
      const CudaVector<float>& d_nextNX,
      const CudaVector<float>& d_nextNY,
            CudaVector<float>& d_bcMX,
            CudaVector<float>& d_bcMY,
            CudaVector<float>& d_bcNX,
            CudaVector<float>& d_bcNY,
            CudaVector<float>& d_curDirectionX,
            CudaVector<float>& d_curDirectionY);
};

}; // namespace skyline

#endif
