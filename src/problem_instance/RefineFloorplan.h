#ifndef REFINE_FLOORPLAN_H
#define REFINE_FLOORPLAN_H

#include "ProblemInstance.h"

namespace macroplacer
{

class Macro;
class Net;
class Pin;

class Painter;

class RefineFloorplan : public ProblemInstance
{
  public:

    // Constructor
    RefineFloorplan(
      float x_min, float y_min, float x_max, float y_max,
      std::shared_ptr<Painter> painter,
      std::vector<Macro*>& macros,
      std::vector<Net*>& nets,
      std::vector<Pin*>& pins);

    // Override Functions
    void updatePointAndGetGrad(
      const CudaVector<float>& var,
            CudaVector<float>& grad) override;

    void getInitialGrad(
      const CudaVector<float>& initial_var,
            CudaVector<float>& initial_grad) override;

    void clipToFeasibleSolution(CudaVector<float>& var) override;

    void updateParameters() override;

    void solveBgnCbk() override;
    void solveEndCbk(int iter, double runtime, const CudaVector<float>& var) override;

    void iterBgnCbk(int iter) override;
    void iterEndCbk(int iter, double runtime, const CudaVector<float>& var) override;

    bool checkConvergence() const override;

    // Non-Override Functions
    void scaleArea(float scale);

    void setNeedExport(bool flag);

    float getSumOverlap() const;

    float getHpwl() const;

  private:

    int num_pin_;
    int num_net_;
    int num_macro_;
    int num_pair_;

    bool need_export_;

    float x_min_;
    float y_min_;
    float x_max_;
    float y_max_;

    float lambda_;

    float hpwl_;
    float sum_overlap_area_;

    std::vector<Macro*> macro_ptrs_;

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

    CudaVector<float> d_wl_grad_;

    /* Data for Overlap gradient computation */
    CudaVector<int>   d_index_pair_;
    CudaVector<float> d_macro_width_;
    CudaVector<float> d_macro_height_;

    CudaVector<float> d_overlap_area_;
    CudaVector<float> d_overlap_grad_;

    CudaVector<float> d_min_ratio_;
    CudaVector<float> d_max_ratio_;
    CudaVector<float> d_macro_area_;
    CudaVector<float> d_macro_area_original_;

    float area_scale_;
    float sum_macro_area_original_;
    float sum_macro_area_;

    // These are constant
    CudaVector<int>   d_pin2net_;
    CudaVector<int>   d_pin2macro_;
    CudaVector<int>   d_net_start_;
    CudaVector<float> d_net_weight_;

    void updateWidthAndHeight(const CudaVector<float>& var);

    void computeWirelengthSubGrad(
      const CudaVector<float>& var,
            CudaVector<float>& grad);

    void computeOverlapSubGrad(
      const CudaVector<float>& var,
            CudaVector<float>& grad);

    // export to database
    void exportToDb(const CudaVector<float>& macro_pos);

    float computeHpwl();

    void printProgress(int iter) const;
};

}; // namespace macroplacer

#endif
