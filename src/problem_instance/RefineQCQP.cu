#include "cuda_linalg/CudaVectorAlgebra.h"
#include "cuda_linalg/CudaSparseMatrixAlgebra.h"
#include "RefineQCQP.h"
#include "Painter.h"

#include "objects/Macro.h"

namespace macroplacer
{

__global__ void clipCircleToChipBoundaryKernel(
  const int    num_macro,
  const float  x_min,
  const float  y_min,
  const float  x_max,
  const float  y_max,
  const float* macro_radius,
        float* macro_cx,
        float* macro_cy)
{
  const int macro_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(macro_id < num_macro)
  {
    float new_cx = macro_cx[macro_id];
    float new_cy = macro_cy[macro_id];
    float radius = macro_radius[macro_id];

    if(new_cx - radius < x_min)
      new_cx = x_min + radius;
    if(new_cx + radius > x_max)
      new_cx = x_max - radius;

    if(new_cy - radius < y_min)
      new_cy = y_min + radius;
    if(new_cy + radius > y_max)
      new_cy = y_max - radius;

    macro_cx[macro_id] = new_cx;
    macro_cy[macro_id] = new_cy;
  }
}

__global__ void computeCircleOverlapSubGradKernel(
  const int    num_pair,
  const int    num_macro,
  const int*   pair_list,
  const float* macro_cx,
  const float* macro_cy,
  const float* ineq_constraint,
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

    float diff_x = abs(cx2 - cx1);
    float diff_x_square = diff_x * diff_x;

    float diff_y = abs(cy2 - cy1);
    float diff_y_square = diff_y * diff_y;

    float dist_square = diff_x_square + diff_y_square;

    float radius_sum_square = ineq_constraint[pair_id];

    overlap_length[pair_id] = max(0.0, radius_sum_square - dist_square);

    if(dist_square < radius_sum_square)
    {
      float x_sign1 = cx1 < cx2 ? -1.0f : +1.0f;
      float x_sign2 = -x_sign1;

      float y_sign1 = cy1 < cy2 ? -1.0f : +1.0f;
      float y_sign2 = -y_sign1;

      atomicAdd(&(overlap_grad_x[macro_id1]), 2 * x_sign1 * diff_x);
      atomicAdd(&(overlap_grad_x[macro_id2]), 2 * x_sign2 * diff_x);

      atomicAdd(&(overlap_grad_y[macro_id1]), 2 * y_sign1 * diff_y);
      atomicAdd(&(overlap_grad_y[macro_id2]), 2 * y_sign2 * diff_y);
    }
  }
}

std::pair<float, float> 
RefineQCQP::convertToScale(float x, float y) const
{
  float scaled_x = (x - offset_x_) * scale_x_;
  float scaled_y = (y - offset_y_) * scale_y_;
  return {scaled_x, scaled_y};
}

std::pair<float, float> 
RefineQCQP::revertToOriginal(float x, float y) const
{
  float original_x = x / scale_x_ + offset_x_;
  float original_y = y / scale_y_ + offset_y_;
  return {original_x, original_y};
}

RefineQCQP::RefineQCQP(
  float scale_x,
  float scale_y,
  float offset_x,
  float offset_y,
  float x_min, 
  float y_min, 
  float x_max, 
  float y_max,
  std::shared_ptr<Painter> painter,
  std::vector<Macro*>& movable_macros,
  const EigenSMatrix& Lmm,
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf,
  const EigenVector&  ineq_constraint) : ProblemInstance(painter)
{
  x_min_     = x_min;
  y_min_     = y_min;
  x_max_     = x_max;
  y_max_     = y_max;
  scale_x_   = scale_x;
  scale_y_   = scale_y;
  offset_x_  = offset_x;
  offset_y_  = offset_y;
  num_macro_ = movable_macros.size();
  num_pair_  = num_macro_ * (num_macro_ - 1) / 2;
  num_var_   = 2 * num_macro_; // {x, y, aspect_ratio}
  lambda_    = 12.0f;

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

    float rect_area = macro_ptr->getOriginalArea() * scale_x * scale_y;
    float radius = std::sqrt(rect_area / k_pi);

    auto [scaled_x, scaled_y] = convertToScale(macro_ptr->getCx(), macro_ptr->getCy());

    h_solution_[macro_id] = scaled_x;
    h_solution_[macro_id + num_macro_] = scaled_y;

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
  d_Lmf_xf_.resize(num_macro_);
  d_Lmf_yf_.resize(num_macro_);
  d_x_slot_.resize(num_macro_);
  d_y_slot_.resize(num_macro_);

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
  std::vector<float> h_ineq_constraint;
  h_ineq_constraint.resize(num_pair_);

  for(int i = 0; i < num_pair_; i++)
    h_ineq_constraint[i] = ineq_constraint(i);

  d_index_pair_.resize(num_pair_);
  d_overlap_length_.resize(num_pair_);
  d_overlap_grad_.resize(num_var_);
  d_radius_.resize(num_macro_);
  d_ineq_constraint_.resize(num_pair_);

  d_index_pair_ = h_index_pair;
  d_radius_ = h_radius;
  d_ineq_constraint_ = h_ineq_constraint;
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
  int num_thread = 64;
  int num_block_cell = (num_macro_ - 1 + num_thread) / num_thread;

  clipCircleToChipBoundaryKernel<<<num_block_cell, num_thread>>>(
    num_macro_,
    x_min_,
    y_min_,
    x_max_,
    y_max_,
    d_radius_.data(),
    var.data(),
    var.data() + num_macro_);
}

void 
RefineQCQP::updateParameters()
{
  sum_overlap_length_ = computeVectorSum(d_overlap_length_);
}

void
RefineQCQP::printProgress(int iter) const
{
  //printf("Iter: %4d SumOverlap: %8f\n", iter, sum_overlap_length_);
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
  printProgress(iter);
}

bool 
RefineQCQP::checkConvergence() const
{
  return false;
}

void
RefineQCQP::computeQuadraticWirelengthSubGrad(
  const CudaVector<float>& var, 
        CudaVector<float>& wl_grad)
{
  thrust::copy(var.begin(), 
               var.begin() + num_macro_, 
               d_x_slot_.begin());

  thrust::copy(var.begin() + num_macro_, 
               var.end(), 
               d_y_slot_.begin());

  sparseMatrixVectorMult(d_Lmm_, d_x_slot_, d_wl_grad_x_);
  vectorAxpy(1.0f, d_Lmf_xf_, d_wl_grad_x_);

  sparseMatrixVectorMult(d_Lmm_, d_y_slot_, d_wl_grad_y_);
  vectorAxpy(1.0f, d_Lmf_yf_, d_wl_grad_y_);

  thrust::copy(
    d_wl_grad_x_.begin(), 
    d_wl_grad_x_.end(), 
    wl_grad.begin());

  thrust::copy(
    d_wl_grad_y_.begin(), 
    d_wl_grad_y_.end(), 
    wl_grad.begin() + num_macro_);

  vectorMulScalar(-1.0f, wl_grad);
}

void 
RefineQCQP::computeCircleOverlapSubGrad(
  const CudaVector<float>& var, 
        CudaVector<float>& overlap_grad)
{
  int num_thread = 64;
  int num_block_pair = (num_pair_ - 1 + num_thread) / num_thread;

  overlap_grad.fillZero();
  d_overlap_length_.fillZero();

  computeCircleOverlapSubGradKernel<<<num_block_pair, num_thread>>>(
    num_pair_,
    num_macro_,
    d_index_pair_.data(),
    var.data(),                /* cx */
    var.data() + num_macro_,   /* cy */
    d_ineq_constraint_.data(),
    d_overlap_length_.data(),
    overlap_grad.data(),               /* overlap_grad_x */
    overlap_grad.data() + num_macro_); /* overlap_grad_y */
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
    auto [new_cx, new_cy] = revertToOriginal(macro_cx, macro_cy);
    macro_ptr->setCx(new_cx);
    macro_ptr->setCy(new_cy);
  }
}

}; // namespace macroplacer
