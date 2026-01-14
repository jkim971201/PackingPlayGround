#ifndef SDP_SOLVER_CPU_H
#define SDP_SOLVER_CPU_H

#include <vector>
#include <memory>
#include "EigenDef.h"

namespace sdp_solver
{

using namespace cuda_linalg;

class SDPInstance;

class SDPSolverCPU
{
  public:

    SDPSolverCPU(std::shared_ptr<SDPInstance> sdp_inst);

    void invertObjectiveSign();

    EigenDMatrix solve();

    std::vector<std::vector<double>> getResult() const;

    double getObjectiveValue() const;

  private:

    void setObjective(EigenSMatrix& C);
    void addEqualityConstraint(EigenSMatrix& Ai, double bi);
    void addInequalityConstraint(EigenSMatrix& Gi, double hi);

    void decompSVD(EigenDMatrix& matrix);

    int n_; // X is n by n matrix
    int m_; // Num Equality   Constraints
    int p_; // Num Inequality Constraints

    double primal_obj_;

    /* ------------ CPU Data ------------ */
    std::vector<std::vector<double>> solution_;
    std::vector<double> b_;
    std::vector<double> h_;
    EigenSMatrix C_;
    std::vector<EigenSMatrix> A_;
    std::vector<EigenSMatrix> G_;
    /* ---------------------------------- */
};

}

#endif
