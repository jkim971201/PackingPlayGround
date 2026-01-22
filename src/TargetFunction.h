#ifndef TARGET_FUNCTION_H
#define TARGET_FUNCTION_H

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

class TargetFunction
{
  public:

    // Constructor
    TargetFunction(
      float x_min, float y_min, float x_max, float y_max,
      std::shared_ptr<Painter> painter,
      std::vector<Macro*>& macros,
      std::vector<Net*>& nets,
      std::vector<Pin*>& pins);

    // APIs
    void updatePointAndGetGrad(
      const CudaVector<float>& cur_x,
      const CudaVector<float>& cur_y,
            CudaVector<float>& grad_x,
            CudaVector<float>& grad_y);

    void getInitialGrad(
      const CudaVector<float>& initial_x,
      const CudaVector<float>& initial_y,
            CudaVector<float>& initial_grad_x,
            CudaVector<float>& initial_grad_y);

    void clipToChipBoundary(
      CudaVector<float>& cell_cx,
      CudaVector<float>& cell_cy);

    void exportToSolver(
      CudaVector<float>& cell_cx,
      CudaVector<float>& cell_cy); 
    // This will be used only to get initial solution

    void updateParameters();

    void solveBgnCbk();
    void solveEndCbk(
      int iter, double runtime,
      const CudaVector<float>& d_cell_x, 
      const CudaVector<float>& d_cell_y);

    void iterBgnCbk(int iter);
    void iterEndCbk(
      int iter, double runtime,
      const CudaVector<float>& d_cell_x, 
      const CudaVector<float>& d_cell_y);

    bool checkConvergence() const;

    int getNumVariable() const;

  private:

    int num_pin_;
    int num_net_;
    int num_macro_;
    int num_pair_;

    bool plot_mode_;

    float x_min_;
    float y_min_;
    float x_max_;
    float y_max_;

    float lambda_;

    float hpwl_;
    float sum_overlap_area_;

    std::vector<float> h_macro_cx_;
    std::vector<float> h_macro_cy_;

    std::vector<Macro*> macro_ptrs_;

    std::shared_ptr<Painter> painter_;

    /* Data for Wirelength gradient computation */
    CudaVector<int>   d_is_max_pin_x_;
    CudaVector<int>   d_is_min_pin_x_;
    CudaVector<int>   d_is_max_pin_y_;
    CudaVector<int>   d_is_min_pin_y_;

    CudaVector<float> d_net_bbox_width_;
    CudaVector<float> d_net_bbox_height_;

    CudaVector<float> d_pin_x_;
    CudaVector<float> d_pin_y_;

    CudaVector<float> d_pin_x_offset_;
    CudaVector<float> d_pin_y_offset_;

    CudaVector<float> d_pin_grad_x_;
    CudaVector<float> d_pin_grad_y_;

    CudaVector<float> d_wl_grad_x_;
    CudaVector<float> d_wl_grad_y_;

    /* Data for Overlap gradient computation */
    CudaVector<int>   d_index_pair_;
    CudaVector<float> d_macro_width_;
    CudaVector<float> d_macro_height_;

    CudaVector<float> d_overlap_area_;
    CudaVector<float> d_overlap_grad_x_;
    CudaVector<float> d_overlap_grad_y_;

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

    void computeOverlapSubGrad(
      const CudaVector<float>& cur_x,
      const CudaVector<float>& cur_y,
            CudaVector<float>& grad_x,
            CudaVector<float>& grad_y);

    // export to database
    void exportToDb(
      const CudaVector<float>& macro_cx, 
      const CudaVector<float>& macro_cy);

    float computeHpwl();

    void printProgress(int iter) const;
};

}; // namespace macroplacer

#endif
