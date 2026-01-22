#ifndef TARGET_FUNCTION_H
#define TARGET_FUNCTION_H

#include "cuda_linalg/CudaVector.h"

namespace macroplacer
{

class Macro;
class Net;
class Pin;

using namespace cuda_linalg;

class TargetFunction
{
  public:

    // Constructor
    TargetFunction(
      std::vector<Macro*>& macros,
      std::vector<Net*>& nets,
      std::vector<Pin*>& pins);

    // APIs
    void updatePointAndGetGrad(
      const CudaVector<float>& cur_x,
      const CudaVector<float>& cur_y,
            CudaVector<float>& grad_x,
            CudaVector<float>& grad_y);

  private:

    int num_pin_;
    int num_net_;
    int num_macro_;

    /* Data for Wirelength gradient computation */
    CudaVector<int>   d_is_max_pin_x_;
    CudaVector<int>   d_is_min_pin_x_;
    CudaVector<int>   d_is_max_pin_y_;
    CudaVector<int>   d_is_min_pin_y_;

    CudaVector<float> d_pin_x_;
    CudaVector<float> d_pin_y_;

    CudaVector<float> d_pin_x_offset_;
    CudaVector<float> d_pin_y_offset_;

    CudaVector<float> d_pin_grad_x_;
    CudaVector<float> d_pin_grad_y_;

    CudaVector<float> d_wl_grad_x_;
    CudaVector<float> d_wl_grad_y_;

    // These are constant
    CudaVector<int>   d_pin2net_;
    CudaVector<int>   d_pin2macro_;
    CudaVector<int>   d_net_start_;
    CudaVector<float> d_net_weight_;

    void computeWirelengthSubGrad(
      const CudaVector<float>& cur_x,
      const CudaVector<float>& cur_y,
            CudaVector<float>& grad_x,
            CudaVector<float>& grad_y);

    /* Data for Overlap gradient computation */
    void computeOverlapSubGrad(
      const CudaVector<float>& cur_x,
      const CudaVector<float>& cur_y,
            CudaVector<float>& grad_x,
            CudaVector<float>& grad_y);

    // These are constant
    CudaVector<int>   d_index_pair_;
    CudaVector<float> d_macro_width_;
    CudaVector<float> d_macro_height_;
};

}; // namespace macroplacer

#endif
