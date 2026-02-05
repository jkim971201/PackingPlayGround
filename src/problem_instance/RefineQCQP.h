#ifndef REFINE_QCQP_H
#define REFINE_QCQP_H

#include "ProblemInstance.h"
#include "EigenDef.h"
#include "cuda_linalg/CudaVector.h"
#include "cuda_linalg/CudaSparseMatrix.h"

namespace macroplacer
{

class Macro;

class RefineQCQP : public ProblemInstance
{
  public:

    // Constructor
    RefineQCQP(
      std::shared_ptr<Painter> painter,
      std::vector<Macro*>& macros,
      const EigenSMatrix& Lmm,
      const EigenVector& Lmf_xf,
      const EigenVector& Lmf_yf);

    // APIs
    // var : {x_vector, y_vector, ... }
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

  private:

    int num_macro_;
    int num_pair_;
    float lambda_;

    std::vector<Macro*> macro_ptrs_;

    /* Data for Wirelength gradient computation */
    CudaVector<float> d_Lmf_xf_;
    CudaVector<float> d_Lmf_yf_;
    CudaSparseMatrix<float> d_Lmm_;

    CudaVector<float> d_wl_grad_x_;
    CudaVector<float> d_wl_grad_y_;
    CudaVector<float> d_wl_grad_;

    float quadratic_wirelength_;

    /* Data for Overlap gradient computation */
    CudaVector<int>   d_index_pair_;
    CudaVector<float> d_radius_;
    CudaVector<float> d_overlap_length_;
    CudaVector<float> d_overlap_grad_;

    float sum_overlap_length_;

    void computeQuadraticWirelengthSubGrad(
      const CudaVector<float>& var,
            CudaVector<float>& grad);

    void computeCircleOverlapSubGrad(
      const CudaVector<float>& var,
            CudaVector<float>& grad);

    // export to database
    void exportToDb(const CudaVector<float>& macro_pos);

    void printProgress(int iter) const;
};

}; // namespace macroplacer

#endif
