#include "cuda_linalg/CudaVectorAlgebra.h"
#include "cuda_linalg/CudaSparseMatrixAlgebra.h"
#include "RefineQCQP.h"
#include "Painter.h"

#include "objects/Macro.h"

namespace macroplacer
{

__global__ void computeCircleOverlapSubGradKernel(
  const int    num_pair,
  const int    num_macro,
  const int*   pair_list,
  const float* macro_cx,
  const float* macro_cy,
  const float* circle_radius,
        float* overlap_length,
        float* overlap_grad_x,
        float* overlap_grad_y)
{
  const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(pair_id < num_pair)
  {
    int pair = pair_list[pair_id];
    int macro_id1 = pair % num_macro;
    int macro_id2 = pair / num_macro;
    assert(macro_id1 < macro_id2);
    
    float cx1 = macro_cx[macro_id1];
    float cy1 = macro_cy[macro_id1];
    float cx2 = macro_cx[macro_id2];
    float cy2 = macro_cy[macro_id2];

    float radius1 = circle_radius[macro_id1];
    float radius2 = circle_radius[macro_id2];
  }
}

RefineQCQP::RefineQCQP(
  std::shared_ptr<Painter> painter,
  std::vector<Macro*>& movable_macros,
  const EigenSMatrix& Lmm,
  const EigenVector& Lmf_xf,
  const EigenVector& Lmf_yf) : ProblemInstance(painter)
{
  num_macro_   = movable_macros.size();
  num_pair_    = num_macro_ * (num_macro_ - 1) / 2;
  num_var_     = 2 * num_macro_; // {x, y, aspect_ratio}
  lambda_      = 1.0f;

  h_solution_.resize(num_var_); 

  std::vector<int> h_index_pair;
  std::vector<float> h_radius;

  h_index_pair.reserve(num_pair_);
  h_radius.reserve(num_macro_);

  macro_ptrs_.reserve(num_macro_);

  constexpr double k_pi = std::numbers::pi;

  for(int macro_id = 0; macro_id < num_macro_; macro_id++)
  {
    auto macro_ptr = movable_macros[macro_id];
    macro_ptrs_.push_back(macro_ptr);

    float rect_area = macro_ptr->getOriginalArea();
    float radius = std::sqrt(rect_area / k_pi);

    h_solution_[macro_id] = macro_ptr->getCx();
    h_solution_[macro_id + num_macro_] = macro_ptr->getCy();

    h_radius.push_back(radius);
    
    for(int macro_id2 = macro_id + 1; macro_id2 < num_macro_; macro_id2++)
      h_index_pair.push_back(macro_id + macro_id2 * num_macro_);
  }

  // Initialize CUDA Kernel
  // wirelength
  d_wl_grad_.resize(num_var_);
  d_wl_grad_x_.resize(num_macro_);
  d_wl_grad_y_.resize(num_macro_);
 
  d_Lmm_.initialize(Lmm);

  std::vector<float> h_Lmf_xf(num_macro_);
  std::vector<float> h_Lmf_yf(num_macro_);

  for(int i = 0; i < num_macro_; i++)
  {
    h_Lmf_xf[i] = Lmf_xf(i);
    h_Lmf_yf[i] = Lmf_yf(i);
  }

  d_Lmf_xf_ = h_Lmf_xf;
  d_Lmf_yf_ = h_Lmf_yf;

  // overlap
  d_index_pair_.resize(num_pair_);
  d_overlap_length_.resize(num_pair_);
  d_overlap_grad_.resize(num_var_);
  d_radius_.resize(num_macro_);

  d_index_pair_ = h_index_pair;
  d_radius_ = h_radius;
}

void
RefineQCQP::updatePointAndGetGrad(const CudaVector<float>& var, CudaVector<float>& grad)
{
  // For WireLength Grad
  computeQuadraticWirelengthSubGrad(var, d_wl_grad_);

  // For Overlap SubGrad
  computeCircleOverlapSubGrad(var, d_overlap_grad_);

  vectorAdd(1.0f, lambda_, d_wl_grad_, d_overlap_grad_, grad);
}

void 
RefineQCQP::getInitialGrad(
  const CudaVector<float>& initial_var,
        CudaVector<float>& initial_grad)
{
  updatePointAndGetGrad(initial_var, initial_grad);
}

void 
RefineQCQP::clipToFeasibleSolution(CudaVector<float>& var)
{

}

void 
RefineQCQP::updateParameters()
{

}

void
RefineQCQP::printProgress(int iter) const
{
}

void 
RefineQCQP::solveBgnCbk()
{
}

void 
RefineQCQP::solveEndCbk(int iter, double runtime, const CudaVector<float>& macro_pos)
{
  exportToDb(macro_pos);
}

void 
RefineQCQP::iterBgnCbk(int iter)
{

}

void 
RefineQCQP::iterEndCbk(
  int iter, double runtime,
  const CudaVector<float>& macro_pos)
{
  updateParameters();
  //printProgress(iter);
}

bool 
RefineQCQP::checkConvergence() const
{
  return false;
}

void
RefineQCQP::computeQuadraticWirelengthSubGrad(
  const CudaVector<float>& var, 
        CudaVector<float>& grad)
{
  sparseMatrixVectorMult(d_Lmm_, var, d_wl_grad_x_);
  vectorAxpy(1.0f, d_Lmf_xf_, d_wl_grad_x_);

  sparseMatrixVectorMult(d_Lmm_, var, d_wl_grad_y_);
  vectorAxpy(1.0f, d_Lmf_yf_, d_wl_grad_y_);

  thrust::copy(
    d_wl_grad_x_.begin(), 
    d_wl_grad_x_.end(), 
    grad.begin());

  thrust::copy(
    d_wl_grad_y_.begin(), 
    d_wl_grad_y_.end(), 
    grad.begin() + num_macro_);
}

void 
RefineQCQP::computeCircleOverlapSubGrad(
  const CudaVector<float>& var, 
        CudaVector<float>& grad)
{
  int num_thread = 64;
  int num_block_pair = (num_pair_ - 1 + num_thread) / num_thread;

  computeCircleOverlapSubGradKernel<<<num_block_pair, num_thread>>>(
    num_pair_,
    num_macro_,
    d_index_pair_.data(),
    var.data(),                /* cx */
    var.data() + num_macro_,   /* cy */
    d_radius_.data(),
    d_overlap_length_.data(),
    grad.data(),               /* overlap_grad_x */
    grad.data() + num_macro_); /* overlap_grad_y */
}

void
RefineQCQP::exportToDb(const CudaVector<float>& d_solution)
{
  thrust::copy(d_solution.begin(), d_solution.end(), h_solution_.begin());

  for(int macro_id = 0; macro_id < num_macro_; macro_id++)
  {
    auto macro_ptr = macro_ptrs_[macro_id];
    float macro_cx = h_solution_[macro_id];
    float macro_cy = h_solution_[macro_id + num_macro_];
    macro_ptr->setCx(macro_cx);
    macro_ptr->setCy(macro_cy);
  }
}

}; // namespace macroplacer
