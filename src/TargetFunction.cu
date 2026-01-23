#include "cuda_linalg/CudaVectorAlgebra.h"
#include "TargetFunction.h"
#include "Painter.h"

#include "objects/Macro.h"
#include "objects/Pin.h"
#include "objects/Net.h"

namespace macroplacer
{

__device__ inline float getXInsideChip(
  const float cell_cx, 
  const float cell_width, 
  const float x_min,
  const float x_max)
{
  float new_cx = cell_cx;
  if(cell_cx - cell_width / 2 < x_min)
    new_cx = x_min + cell_width / 2;
  if(cell_cx + cell_width / 2 > x_max)
    new_cx = x_max - cell_width / 2;
  return new_cx;
}

__device__ inline float getYInsideChip(
  const float cell_cy, 
  const float cell_height, 
  const float y_min,  
  const float y_max)
{
  float new_cy = cell_cy;
  if(cell_cy - cell_height / 2 < y_min)
    new_cy = y_min + cell_height / 2;
  if(cell_cy + cell_height / 2 > y_max)
    new_cy = y_max - cell_height / 2;
  return new_cy;
}

__global__ void clipToChipBoundaryKernel(
  const int    num_cell,
  const float  x_min,
  const float  y_min,
  const float  x_max,
  const float  y_max,
  const float* cell_width,
  const float* cell_height,
        float* cell_cx,
        float* cell_cy)
{
  const int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(cell_id < num_cell)
  {
    cell_cx[cell_id] = getXInsideChip(cell_cx[cell_id], cell_width[cell_id], x_min, x_max);
    cell_cy[cell_id] = getYInsideChip(cell_cy[cell_id], cell_height[cell_id], y_min, y_max);
  }
}

__global__ void updatePinCoordinateKernel(
  const int    num_pin, 
  const int*   pin2cell,
  const float* new_cell_x,
  const float* new_cell_y, 
  const float* pin_offset_x,
  const float* pin_offset_y,
        float* pin_x,
        float* pin_y)
{
  const int pin_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(pin_id < num_pin)
  {
    int cell_id = pin2cell[pin_id];
    // Some pins are from bterms (IO pins), 
    // so we should check if cell_id is valid.
    if(cell_id < 0)
      return;
    pin_x[pin_id] = new_cell_x[cell_id] + pin_offset_x[pin_id];
    pin_y[pin_id] = new_cell_y[cell_id] + pin_offset_y[pin_id];
  }
}

__global__ void identifyMinMax(
  const int    num_pin,
  const int*   net_start,
  const int*   pin2net,
  const float* pin_x, 
  const float* pin_y, 
        int*   is_max_x_arr,
        int*   is_min_x_arr,
        int*   is_max_y_arr,
        int*   is_min_y_arr)
{
  const int pin_id1 = blockIdx.x * blockDim.x + threadIdx.x;

  if(pin_id1 < num_pin)
  {
    int is_max_x = 1;
    int is_min_x = 1;
    int is_max_y = 1;
    int is_min_y = 1;

    float x_this_pin = pin_x[pin_id1];
    float y_this_pin = pin_y[pin_id1];

    const int net_id = pin2net[pin_id1];
    for(int pin_id2 = net_start[net_id]; pin_id2 < net_start[net_id+1]; pin_id2++)
    {
      if(pin_id1 == pin_id2)
        continue;
      else
      {
        float x_othter_pin = pin_x[pin_id2];
        float y_othter_pin = pin_y[pin_id2];
        if(x_othter_pin >= x_this_pin)
          is_max_x = 0;
        if(x_othter_pin <= x_this_pin)
          is_min_x = 0;
        if(y_othter_pin >= y_this_pin)
          is_max_y = 0;
        if(y_othter_pin <= y_this_pin)
          is_min_y = 0;
      }
    }

    is_max_x_arr[pin_id1] = is_max_x;
    is_min_x_arr[pin_id1] = is_min_x;

    assert(is_max_x + is_min_x <= 1);

    is_max_y_arr[pin_id1] = is_max_y;
    is_min_y_arr[pin_id1] = is_min_y;

    assert(is_max_y + is_min_y <= 1);
  }
}

__global__ void computePinGrad(
  const int    num_pin,
  const int*   pin2net,
  const int*   is_max_pin_x,
  const int*   is_min_pin_x,
  const int*   is_max_pin_y,
  const int*   is_min_pin_y,
  const float* net_weight,
        float* pin_grad_x,  
        float* pin_grad_y)
{
  const int pin_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(pin_id < num_pin)
  {
    int net_id = pin2net[pin_id]; 
    float weight_this_net = net_weight[net_id];
    float pull_value = weight_this_net * 1.0f;

    if(is_max_pin_x[pin_id] == 1)
      pin_grad_x[pin_id] = -pull_value;
    else if(is_min_pin_x[pin_id] == 1)
      pin_grad_x[pin_id] = +pull_value;

    if(is_max_pin_y[pin_id] == 1)
      pin_grad_y[pin_id] = -pull_value;
    else if(is_min_pin_y[pin_id] == 1)
      pin_grad_y[pin_id] = +pull_value;
  }
}

__global__ void addPinGrad(
  const int    num_pin,
  const int*   pin2cell,
  const float* pin_grad_x,
  const float* pin_grad_y,
        float* cell_grad_x,
        float* cell_grad_y)
{
  const int pin_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(pin_id < num_pin)
  {
    int cell_id = pin2cell[pin_id];
    if(cell_id < 0)
      return;

    float grad_x = pin_grad_x[pin_id];
    float grad_y = pin_grad_y[pin_id];
    atomicAdd(&cell_grad_x[cell_id], grad_x);
    atomicAdd(&cell_grad_y[cell_id], grad_y);
  }
}

__global__ void computeNetBBoxKernel(
  const int    num_net,
  const int*   net_start,
  const float* pin_x,
  const float* pin_y,
        float* net_bbox_width,
        float* net_bbox_height)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_net)
  {
    float max_pin_x = 0;
    float max_pin_y = 0;
    float min_pin_x = k_float_max;
    float min_pin_y = k_float_max;

    for(int j = net_start[i]; j < net_start[i+1]; j++)
    {
      max_pin_x = max(pin_x[j], max_pin_x);
      min_pin_x = min(pin_x[j], min_pin_x);
      max_pin_y = max(pin_y[j], max_pin_y);
      min_pin_y = min(pin_y[j], min_pin_y);
    }
 
    net_bbox_width[i] = max_pin_x - min_pin_x;
    net_bbox_height[i] = max_pin_y - min_pin_y;
  }
}

__global__ void computeOverlapSubGradKernel(
  const int    num_pair,
  const int    num_macro,
  const int*   pair_list,
  const float* macro_cx,
  const float* macro_cy,
  const float* macro_width,
  const float* macro_height,
        float* overlap_area,
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

    float width1  = macro_width[macro_id1];
    float width2  = macro_width[macro_id2];
    float height1 = macro_height[macro_id1];
    float height2 = macro_height[macro_id2];

    float x_min1 = cx1 - width1  / 2.0;
    float x_max1 = cx1 + width1  / 2.0;
    float y_min1 = cy1 - height1 / 2.0;
    float y_max1 = cy1 + height1 / 2.0;

    float x_min2 = cx2 - width2  / 2.0;
    float x_max2 = cx2 + width2  / 2.0;
    float y_min2 = cy2 - height2 / 2.0;
    float y_max2 = cy2 + height2 / 2.0;

    float overlap_length_x = 
      max(0.0f, min(x_max1, x_max2) - max(x_min1, x_min2));

    float overlap_length_y = 
      max(0.0f, min(y_max1, y_max2) - max(y_min1, y_min2));

    float overlap_rect_area = overlap_length_x * overlap_length_y;
    overlap_area[pair_id] = overlap_rect_area;

    if(overlap_rect_area > 0.0f)
    {
      float x_sign1 = cx1 < cx2 ? -1.0f : +1.0f;
      float x_sign2 = -x_sign1;

      float y_sign1 = cy1 < cy2 ? -1.0f : +1.0f;
      float y_sign2 = -y_sign1;

      atomicAdd(&(overlap_grad_x[macro_id1]), x_sign1 * overlap_length_x);
      atomicAdd(&(overlap_grad_x[macro_id2]), x_sign2 * overlap_length_x);

      atomicAdd(&(overlap_grad_y[macro_id1]), y_sign1 * overlap_length_y);
      atomicAdd(&(overlap_grad_y[macro_id2]), y_sign2 * overlap_length_y);
    }
  }
}

TargetFunction::TargetFunction(
  float x_min, float y_min, float x_max, float y_max,
  std::shared_ptr<Painter> painter,
  std::vector<Macro*>& macros,
  std::vector<Net*>& nets,
  std::vector<Pin*>& pins)
{
  painter_   = painter;
  x_min_     = x_min;
  y_min_     = y_min;
  x_max_     = x_max;
  y_max_     = y_max;
  num_net_   = nets.size();
  num_pin_   = pins.size();
  num_macro_ = macros.size();
  num_pair_  = num_macro_ * (num_macro_ - 1) / 2;
  num_var_   = 2 * num_macro_;

  lambda_    = 5.0f;

  h_macro_pos_.resize(num_var_);
  // We don't want to store these data permanently
  std::vector<int> h_pin2net(num_pin_);
  std::vector<int> h_pin2macro(num_pin_, -1);
  std::vector<int> h_net_start(num_net_+ 1);
  std::vector<float> h_macro_width(num_macro_);
  std::vector<float> h_macro_height(num_macro_);
  std::vector<int> h_index_pair;
  h_index_pair.reserve(num_pair_);

  for(auto net_ptr : nets)
  {
    auto net_pins = net_ptr->getPins();
    int net_id = net_ptr->id();
    h_net_start[net_id] = net_pins[0].id();
    for(auto& pin : net_pins)
      h_pin2net[pin.id()] = net_id;
  }

  h_net_start[num_net_] = num_pin_;

  macro_ptrs_.reserve(num_macro_);
  for(int macro_id = 0; macro_id < num_macro_; macro_id++)
  {
    auto macro_ptr = macros[macro_id];
    macro_ptrs_.push_back(macro_ptr);
    h_macro_pos_[macro_id] = macro_ptr->getCx();
    h_macro_pos_[macro_id + num_macro_] = macro_ptr->getCy();
    h_macro_width[macro_id] = macro_ptr->getWidth();
    h_macro_height[macro_id] = macro_ptr->getHeight();
    for(auto& pin : macro_ptr->getPins())
      h_pin2macro[pin->id()] = macro_id;

    for(int macro_id2 = macro_id + 1; macro_id2 < num_macro_; macro_id2++)
      h_index_pair.push_back(macro_id + macro_id2 * num_macro_);
  }

  // Initialize CUDA Kernel
  // wirelength
  d_is_max_pin_x_.resize(num_pin_);
  d_is_min_pin_x_.resize(num_pin_);
  d_is_max_pin_y_.resize(num_pin_);
  d_is_min_pin_y_.resize(num_pin_);

  d_net_bbox_width_.resize(num_net_);
  d_net_bbox_height_.resize(num_net_);

  d_pin_x_.resize(num_pin_);
  d_pin_y_.resize(num_pin_);

  d_pin_x_offset_.resize(num_pin_);
  d_pin_y_offset_.resize(num_pin_);

  d_pin_x_offset_.fillZero();
  d_pin_y_offset_.fillZero();

  d_pin_grad_x_.resize(num_pin_);
  d_pin_grad_y_.resize(num_pin_);

  d_wl_grad_.resize(num_var_);

  d_pin2net_.resize(num_pin_);
  d_pin2macro_.resize(num_pin_);
  d_net_start_.resize(num_net_ + 1);
  d_net_weight_.resize(num_net_);
  
  d_pin2net_   = h_pin2net;
  d_pin2macro_ = h_pin2macro;
  d_net_start_ = h_net_start;

  // overlap
  d_macro_width_.resize(num_macro_);
  d_macro_height_.resize(num_macro_);
  d_index_pair_.resize(num_pair_);

  d_overlap_area_.resize(num_pair_);

  d_overlap_grad_.resize(num_var_);

  d_macro_width_  = h_macro_width;
  d_macro_height_ = h_macro_height;
  d_index_pair_   = h_index_pair;
}

void
TargetFunction::updatePointAndGetGrad(const CudaVector<float>& var, CudaVector<float>& grad)
{
  // For WireLength SubGrad
  computeWirelengthSubGrad(var, d_wl_grad_);

  // For Density SubGrad
  computeOverlapSubGrad(var, d_overlap_grad_);

  vectorAdd(1.0f, lambda_, d_wl_grad_, d_overlap_grad_, grad);
}

void 
TargetFunction::getInitialGrad(
  const CudaVector<float>& initial_var,
        CudaVector<float>& initial_grad)
{
  computeWirelengthSubGrad(initial_var, d_wl_grad_);

  computeOverlapSubGrad(initial_var, d_overlap_grad_);

  vectorAdd(1.0f, lambda_, d_wl_grad_, d_overlap_grad_, initial_grad);
}

void 
TargetFunction::clipToChipBoundary(CudaVector<float>& var)
{
  int num_thread = 64;
  int num_block_cell = (num_macro_ - 1 + num_thread) / num_thread;

  clipToChipBoundaryKernel<<<num_block_cell, num_thread>>>(
    num_macro_,
    x_min_,
    y_min_,
    x_max_,
    y_max_,
    d_macro_width_.data(),
    d_macro_height_.data(),
    var.data(),
    var.data() + num_macro_);
}

void 
TargetFunction::updateParameters()
{
  hpwl_ = computeHpwl();
  sum_overlap_area_ = computeVectorSum(d_overlap_area_);
}

void
TargetFunction::printProgress(int iter) const
{
  if(iter == 0 || iter % 1000 == 0)
  {
    printf("Iter: %4d HPWL: %8f SumOverlap: %8f\n", 
      iter, hpwl_, sum_overlap_area_);
  }
}

void 
TargetFunction::solveBgnCbk()
{
}

void 
TargetFunction::solveEndCbk(int iter, double runtime, const CudaVector<float>& macro_pos)
{
  exportToDb(macro_pos);
}

void 
TargetFunction::iterBgnCbk(int iter)
{
  updateParameters();
  printProgress(iter);
}

void 
TargetFunction::iterEndCbk(
  int iter, double runtime,
  const CudaVector<float>& macro_pos)
{

}

bool 
TargetFunction::checkConvergence() const
{
  return false;
}

int 
TargetFunction::getNumVariable() const
{
  return num_var_;
}

void
TargetFunction::computeWirelengthSubGrad(const CudaVector<float>& pos, CudaVector<float>& wl_grad)
{
  int num_thread    = 64;
  int num_block_pin = (num_pin_ + num_thread) / num_thread; 
  int num_block_net = (num_net_ + num_thread) / num_thread; 

  wl_grad.fillZero();

  // Step #1: Update PinCoordinate
  updatePinCoordinateKernel<<<num_block_pin, num_thread>>>(
    num_pin_, 
    d_pin2macro_.data(),
    pos.data(),              /* cur_x */
    pos.data() + num_macro_, /* cur_y */
    d_pin_x_offset_.data(),
    d_pin_y_offset_.data(),
    d_pin_x_.data(),
    d_pin_y_.data());

  // Step #2: Identify if each pin is max or pin
  identifyMinMax<<<num_block_net, num_thread>>>(
    num_pin_, 
    d_net_start_.data(), 
    d_pin2net_.data(),
    d_pin_x_.data(),
    d_pin_y_.data(),
    d_is_max_pin_x_.data(),
    d_is_min_pin_y_.data(),
    d_is_max_pin_x_.data(),
    d_is_min_pin_y_.data());

  // Step #3: Compute SubGrad for each pin
  computePinGrad<<<num_block_pin, num_thread>>>(
    num_pin_,
    d_pin2net_.data(),
    d_is_max_pin_x_.data(),
    d_is_min_pin_x_.data(),
    d_is_max_pin_y_.data(),
    d_is_min_pin_y_.data(),
    d_net_weight_.data(),
    d_pin_grad_x_.data(),
    d_pin_grad_y_.data());

  // Step #4: Add PinGrad for each CellGrad
  addPinGrad<<<num_block_pin, num_thread>>>(
    num_pin_, 
    d_pin2macro_.data(),
    d_pin_grad_x_.data(),
    d_pin_grad_y_.data(),
    wl_grad.data(),               /* grad_x */
    wl_grad.data() + num_macro_); /* grad_y */
}

void 
TargetFunction::computeOverlapSubGrad(
  const CudaVector<float>& pos, 
        CudaVector<float>& overlap_grad)
{
  int num_thread = 64;
  int num_block_pair = (num_pair_ - 1 + num_thread) / num_thread;

  d_overlap_area_.fillZero();
  overlap_grad.fillZero();

  computeOverlapSubGradKernel<<<num_block_pair, num_thread>>>(
    num_pair_,
    num_macro_,
    d_index_pair_.data(),
    pos.data(),              /* cur_x */
    pos.data() + num_macro_, /* cur_y */
    d_macro_width_.data(),
    d_macro_height_.data(),
    d_overlap_area_.data(),
    overlap_grad.data(),               /* overlap_grad_x */
    overlap_grad.data() + num_macro_); /* overlap_grad_y */
}

float
TargetFunction::computeHpwl()
{
  d_net_bbox_width_.fillZero();
  d_net_bbox_height_.fillZero();

  int num_thread = 64;
  int num_block_net = (num_net_ - 1 + num_thread) / num_thread;

  computeNetBBoxKernel<<<num_block_net, num_thread>>>(
    num_net_,
    d_net_start_.data(),
    d_pin_x_.data(),
    d_pin_y_.data(),
    d_net_bbox_width_.data(),
    d_net_bbox_height_.data());

  float hpwl_x = computeVectorSum(d_net_bbox_width_);
  float hpwl_y = computeVectorSum(d_net_bbox_height_);
  return hpwl_x + hpwl_y;
}

void
TargetFunction::exportToSolver(CudaVector<float>& var_from_solver)
{
  // Host to Device
  var_from_solver = h_macro_pos_;
}

void
TargetFunction::exportToDb(const CudaVector<float>& macro_pos)
{
  thrust::copy(macro_pos.begin(), macro_pos.end(), h_macro_pos_.begin());

  for(int macro_id = 0; macro_id < num_macro_; macro_id++)
  {
    auto macro_ptr = macro_ptrs_[macro_id];
    float macro_cx = h_macro_pos_[macro_id];
    float macro_cy = h_macro_pos_[macro_id + num_macro_];
    macro_ptr->setCx(macro_cx);
    macro_ptr->setCy(macro_cy);
  }
}

}; // namespace macroplacer
