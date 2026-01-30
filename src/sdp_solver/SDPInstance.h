#ifndef SDP_INSTANCE_H
#define SDP_INSTANCE_H

#include "EigenDef.h"
#include <vector>

namespace sdp_solver
{

class SDPInstance
{
  public:

  cuda_linalg::EigenSMatrix obj_matrix;

  std::vector<double>       eq_const_val;
  std::vector<double>       ineq_const_val;

  std::vector<cuda_linalg::EigenSMatrix> eq_const_matrix;
  std::vector<cuda_linalg::EigenSMatrix> ineq_const_matrix;

  void setObjectiveMatrix(cuda_linalg::EigenSMatrix& matrix)
  {
    obj_matrix = matrix;
  }

  void addEqualityConstraint(cuda_linalg::EigenSMatrix& matrix, double val)
  {
    eq_const_matrix.push_back(matrix);
    eq_const_val.push_back(val);
  }

  void addInequalityConstraint(cuda_linalg::EigenSMatrix& matrix, double val)
  {
    ineq_const_matrix.push_back(matrix);
    ineq_const_val.push_back(val);
  }
};

}

#endif
