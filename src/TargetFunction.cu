#include "cuda_linalg/CudaVectorAlgebra.h"
#include "TargetFunction.h"

#include "objects/Macro.h"
#include "objects/Pin.h"
#include "objects/Net.h"

namespace macroplacer
{

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
        if(x_othter_pin > x_this_pin)
          is_max_x = 0;
        if(x_othter_pin < x_this_pin)
          is_min_x = 0;
        if(y_othter_pin > y_this_pin)
          is_max_y = 0;
        if(y_othter_pin < y_this_pin)
          is_min_y = 0;
      }
    }

    is_max_x_arr[pin_id1] = is_max_x;
    is_min_x_arr[pin_id1] = is_min_x;
    is_max_y_arr[pin_id1] = is_max_y;
    is_min_y_arr[pin_id1] = is_min_y;
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
      pin_grad_x[pin_id] = +pull_value;
    else if(is_min_pin_x[pin_id] == 1)
      pin_grad_x[pin_id] = -pull_value;

    if(is_max_pin_y[pin_id] == 1)
      pin_grad_y[pin_id] = +pull_value;
    else if(is_min_pin_y[pin_id] == 1)
      pin_grad_y[pin_id] = -pull_value;
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

TargetFunction::TargetFunction(
  std::vector<Macro*>& macros,
  std::vector<Net*>& nets,
  std::vector<Pin*>& pins)
{
  num_net_   = nets.size();
  num_pin_   = pins.size();
  num_macro_ = macros.size();

  // We don't want to store these data permanently
  std::vector<int> h_pin2net(num_pin_);
  std::vector<int> h_pin2macro(num_pin_, -1);
  std::vector<int> h_net_start(num_net_+ 1);
  std::vector<float> h_macro_width(num_macro_);
  std::vector<float> h_macro_height(num_macro_);

  for(auto net_ptr : nets)
  {
    auto net_pins = net_ptr->getPins();
    int net_id = net_ptr->id();
    h_net_start[net_id] = net_pins[0].id();
    for(auto& pin : net_pins)
      h_pin2net[pin.id()] = net_id;
  }

  h_net_start[num_net_] = num_pin_;

  for(int macro_id = 0; macro_id < num_macro_; macro_id++)
  {
    auto macro_ptr = macros[macro_id];
    h_macro_width[macro_id] = macro_ptr->getWidth();
    h_macro_height[macro_id] = macro_ptr->getHeight();
    for(auto& pin : macro_ptr->getPins())
      h_pin2macro[pin->id()] = macro_id;
  }

  // Initialize CUDA Kernel
  // wirelength
  d_is_max_pin_x_.resize(num_pin_);
  d_is_min_pin_x_.resize(num_pin_);
  d_is_max_pin_y_.resize(num_pin_);
  d_is_min_pin_y_.resize(num_pin_);

  d_pin_x_.resize(num_pin_);
  d_pin_y_.resize(num_pin_);

  d_pin_x_offset_.resize(num_pin_);
  d_pin_y_offset_.resize(num_pin_);

  d_pin_x_offset_.fillZero();
  d_pin_y_offset_.fillZero();

  d_pin_grad_x_.resize(num_pin_);
  d_pin_grad_y_.resize(num_pin_);

  d_wl_grad_x_.resize(num_pin_);
  d_wl_grad_y_.resize(num_pin_);

  d_pin2net_.resize(num_pin_);
  d_pin2macro_.resize(num_pin_);
  d_net_start_.resize(num_net_ + 1);
  d_net_weight_.resize(num_net_);
  
  thrust::copy(h_pin2net.begin(),   h_pin2net.end(),   d_pin2net_.begin());
  thrust::copy(h_pin2macro.begin(), h_pin2macro.end(), d_pin2macro_.begin());
  thrust::copy(h_net_start.begin(), h_net_start.end(), d_net_start_.begin());

  // overlap
  d_macro_width_.resize(num_macro_);
  d_macro_height_.resize(num_macro_);

  thrust::copy(h_macro_width.begin(), h_macro_width.end(), d_macro_width_.begin());
  thrust::copy(h_macro_height.begin(), h_macro_height.end(), d_macro_height_.begin());
}

void
TargetFunction::updatePointAndGetGrad(
  const CudaVector<float>& cur_x,
  const CudaVector<float>& cur_y,
        CudaVector<float>& grad_x,
        CudaVector<float>& grad_y)
{
  // For WireLength SubGrad
  computeWirelengthSubGrad(cur_x, cur_y, d_wl_grad_x_, d_wl_grad_y_);

  // For Density SubGrad
  computeOverlapSubGrad(cur_x, cur_y, grad_x, grad_y);
}

void
TargetFunction::computeWirelengthSubGrad(
  const CudaVector<float>& cur_x,
  const CudaVector<float>& cur_y,
        CudaVector<float>& wl_grad_x,
        CudaVector<float>& wl_grad_y)
{
  int num_thread    = 64;
  int num_block_pin = (num_pin_ + num_thread) / num_thread; 
  int num_block_net = (num_net_ + num_thread) / num_thread; 

  wl_grad_x.fillZero();
  wl_grad_y.fillZero();

  // Step #1: Update PinCoordinate
  updatePinCoordinateKernel<<<num_block_pin, num_thread>>>(
    num_pin_, 
    d_pin2macro_.data(),
    cur_x.data(),
    cur_y.data(),
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
    wl_grad_x.data(),
    wl_grad_y.data());
}

void 
TargetFunction::computeOverlapSubGrad(
  const CudaVector<float>& cur_x,
  const CudaVector<float>& cur_y,
        CudaVector<float>& grad_x,
        CudaVector<float>& grad_y)
{

}

}; // namespace macroplacer
