#include <cstdio>
#include <memory>
#include <chrono>
#include <cmath>

#include "AdamSolver.h"
#include "TargetFunction.h"

#include "Util.h"
#include "cuda_linalg/CudaVectorAlgebra.h"

namespace macroplacer
{

__global__ void moveForwardKernelAdam(
  const int    num_cell,
  const float  stepLength,
  const float* curX,
  const float* curY,
  const float* curDirectionX,
  const float* curDirectionY,
        float* nextX,
        float* nextY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < num_cell)
  {
    nextX[i] = curX[i] - stepLength * curDirectionX[i];
    nextY[i] = curY[i] - stepLength * curDirectionY[i];
  }
}

__global__ void updateDirectionKernelAdam(
  const int    num_cell,
  const float  epsilon,
  const float* d_ptr_bcMX,
  const float* d_ptr_bcMY,
  const float* d_ptr_bcNX,
  const float* d_ptr_bcNY,
        float* d_ptr_curDirectionX,
        float* d_ptr_curDirectionY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_cell)
  {
    float bcnx = max(0.0, d_ptr_bcNX[i]);
    float bcny = max(0.0, d_ptr_bcNY[i]);
    d_ptr_curDirectionX[i] = d_ptr_bcMX[i] / ( sqrtf(bcnx) + epsilon );
    d_ptr_curDirectionY[i] = d_ptr_bcMY[i] / ( sqrtf(bcny) + epsilon );
  }
}

AdamSolver::AdamSolver(std::shared_ptr<TargetFunction> problem)
  : SolverBase(problem)
{
  alpha_  = 1e-1; // learning_rate
  beta1_  = 0.9;
  beta2_  = 0.999;
  beta1k_ = beta1_;
  beta2k_ = beta2_;
}

void
AdamSolver::initSolver()
{
  initForCUDAKernel();

  target_function_->getInitialGrad(
    d_curX_,
    d_curY_,
    d_curGradX_,
    d_curGradY_);

  target_function_->updateParameters();
}

void
AdamSolver::moveForward(
  const float stepLength,
  const CudaVector<float>& d_curX,
  const CudaVector<float>& d_curY,
  const CudaVector<float>& d_curDirectionX,
  const CudaVector<float>& d_curDirectionY,
        CudaVector<float>& d_nextX,
        CudaVector<float>& d_nextY)
{
  int num_thread = 64;
  int num_block_cell = (num_var_ - 1 + num_thread) / num_thread;

  moveForwardKernelAdam<<<num_block_cell, num_thread>>>(
    num_var_, 
    stepLength,
    d_curX.data(), 
    d_curY.data(), 
    d_curDirectionX.data(),
    d_curDirectionY.data(),
    d_nextX.data(), 
    d_nextY.data() );

  target_function_->clipToChipBoundary(d_nextX, d_nextY);
}

void
AdamSolver::updateMoment(
  const CudaVector<float>& d_curMX,
  const CudaVector<float>& d_curMY,
  const CudaVector<float>& d_curNX,
  const CudaVector<float>& d_curNY,
  const CudaVector<float>& d_curGradX,
  const CudaVector<float>& d_curGradY,
        CudaVector<float>& d_nextMX,
        CudaVector<float>& d_nextMY,
        CudaVector<float>& d_nextNX,
        CudaVector<float>& d_nextNY)
{
  const float one_minus_beta1 = (1.0 - beta1_);
  const float one_minus_beta2 = (1.0 - beta2_);

  vectorAdd(beta1_, -one_minus_beta1, d_curMX, d_curGradX, d_nextMX);
  vectorAdd(beta1_, -one_minus_beta1, d_curMY, d_curGradY, d_nextMY);

  vectorAxpySquare(beta2_, one_minus_beta2, d_curNX, d_curGradX, d_nextNX);
  vectorAxpySquare(beta2_, one_minus_beta2, d_curNY, d_curGradY, d_nextNY);
}

void
AdamSolver::updateDirection(
  const CudaVector<float>& d_nextMX,
  const CudaVector<float>& d_nextMY,
  const CudaVector<float>& d_nextNX,
  const CudaVector<float>& d_nextNY,
        CudaVector<float>& d_bcMX,
        CudaVector<float>& d_bcMY,
        CudaVector<float>& d_bcNX,
        CudaVector<float>& d_bcNY,
        CudaVector<float>& d_curDirectionX,
        CudaVector<float>& d_curDirectionY)
{
  float coeff1 = 1.0 / (1.0 - beta1k_);
  float coeff2 = 1.0 / (1.0 - beta2k_);

  vectorMulScalar(coeff1, d_nextMX, d_bcMX);
  vectorMulScalar(coeff1, d_nextMY, d_bcMY);

  vectorMulScalar(coeff2, d_nextNX, d_bcNX);
  vectorMulScalar(coeff2, d_nextNY, d_bcNY);

  int num_thread = 64;
  int num_block_cell = (num_var_ - 1 + num_thread) / num_thread;

  updateDirectionKernelAdam<<<num_block_cell, num_thread>>>(
    num_var_, 
    epsilon_,
    d_bcMX.data(),
    d_bcMY.data(),
    d_bcNX.data(),
    d_bcNY.data(),
    d_curDirectionX.data(),
    d_curDirectionY.data() );
}

void
AdamSolver::solve()
{
  initSolver();

  printf("Adam Solve Start\n");
  target_function_->solveBgnCbk();

  auto solve_start_chrono = getChronoNow();

  int iter = 0;
  int max_opt_iter = 1000;
  for(; iter < max_opt_iter; iter++)
  {
    target_function_->iterBgnCbk(iter);

    // Step #1: Compute next gradient
    target_function_->updatePointAndGetGrad(
      d_curX_,
      d_curY_,
      d_curGradX_,
      d_curGradY_);

    // Step #2: Update moment
    updateMoment(d_curMX_,
                 d_curMY_,
                 d_curNX_,
                 d_curNY_,
                 d_curGradX_,
                 d_curGradY_,
                 d_nextMX_,
                 d_nextMY_,
                 d_nextNX_,
                 d_nextNY_);

    // Step #3: Update direction
    updateDirection(d_nextMX_,
                    d_nextMY_,
                    d_nextNX_,
                    d_nextNY_,
                    d_bcMX_,
                    d_bcMY_,
                    d_bcNX_,
                    d_bcNY_,
                    d_curDirectionX_,
                    d_curDirectionY_);
    
    // Step #4: Move forward with the direction vector
    moveForward(alpha_,
                d_curX_, 
                d_curY_,
                d_curDirectionX_,
                d_curDirectionY_,
                d_nextX_,
                d_nextY_);

    updateOneIteration(iter);

    if(target_function_->checkConvergence() == true)
      break;

    target_function_->iterEndCbk(
      iter, evalTime(solve_start_chrono), d_curX_, d_curY_);
  }

  target_function_->solveEndCbk(
    iter, evalTime(solve_start_chrono), d_curX_, d_curY_);
}

void
AdamSolver::initForCUDAKernel()
{
  // Step #1. Vectors for Adam
  d_curGradX_.resize(num_var_);
  d_curGradY_.resize(num_var_);

  d_curDirectionX_.resize(num_var_);
  d_curDirectionY_.resize(num_var_);

  d_curMX_.resize(num_var_);
  d_curMY_.resize(num_var_);

  d_nextMX_.resize(num_var_);
  d_nextMY_.resize(num_var_);

  d_curNX_.resize(num_var_);
  d_curNY_.resize(num_var_);

  d_nextNX_.resize(num_var_);
  d_nextNY_.resize(num_var_);

  d_bcMX_.resize(num_var_);
  d_bcMY_.resize(num_var_);

  d_bcNX_.resize(num_var_);
  d_bcNY_.resize(num_var_);

  beta1k_ = beta1_;
  beta2k_ = beta2_;

  epsilon_ = 1e-8;

  // Step #2. Set Initial Solution
  setInitialSolution();
}

void
AdamSolver::setInitialSolution()
{
  // Host -> Device
  target_function_->exportToSolver(d_curX_, d_curY_);
}

void
AdamSolver::updateOneIteration(int iter)
{
  // Current <= Next
  d_curX_.swap(d_nextX_);
  d_curY_.swap(d_nextY_);

  d_curMX_.swap(d_nextMX_);
  d_curMY_.swap(d_nextMY_);

  d_curNX_.swap(d_nextNX_);
  d_curNY_.swap(d_nextNY_);

  alpha_ *= 0.997;

  beta1k_ *= beta1_;
  beta2k_ *= beta2_;
}

}; 
