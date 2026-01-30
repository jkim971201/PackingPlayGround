#include <cstdio>
#include <iostream>

#include <Eigen/Dense>

#include "SDPInstance.h"
#include "SDPSolverCPU.h"
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

std::shared_ptr<ndarray<int,1>>     nint(const std::vector<int>    &X) { return new_array_ptr<int>(X);    }
std::shared_ptr<ndarray<double,1>>  ndou(const std::vector<double> &X) { return new_array_ptr<double>(X); }

Matrix::t eigen2fusion(const cuda_linalg::EigenSMatrix& mat)
{
	int N   = mat.rows();
	int NNZ = mat.nonZeros();

	std::vector<int>    rows(NNZ);
	std::vector<int>    cols(NNZ);
	std::vector<double> vals(NNZ);

	int idx = 0;
	for(int i = 0; i < mat.outerSize(); i++)
	{
		for(cuda_linalg::EigenSMatrix::InnerIterator it(mat, i); it; ++it)
		{
			rows[idx] = it.row();
			cols[idx] = it.col();
			vals[idx] = static_cast<double>(it.value());
			idx++;
		}
	}

	return Matrix::sparse(N, N, nint(rows), nint(cols), ndou(vals) );
}

namespace sdp_solver
{

SDPSolverCPU::SDPSolverCPU(std::shared_ptr<SDPInstance> sdp_inst)
{
  m_ = 0;
  p_ = 0;

  int num_eq_constr = sdp_inst->eq_const_val.size();
  int num_ineq_constr = sdp_inst->ineq_const_val.size();

  setObjective(sdp_inst->obj_matrix);

  auto& eq_const_matrix   = sdp_inst->eq_const_matrix;
  auto& eq_const_val      = sdp_inst->eq_const_val;

  auto& ineq_const_matrix = sdp_inst->ineq_const_matrix;
  auto& ineq_const_val    = sdp_inst->ineq_const_val;

  for(int i = 0; i < num_eq_constr; i++)
    addEqualityConstraint(eq_const_matrix[i], eq_const_val[i]);

  for(int i = 0; i < num_ineq_constr; i++)
    addInequalityConstraint(ineq_const_matrix[i], ineq_const_val[i]);
}

void
SDPSolverCPU::setObjective(EigenSMatrix& C)
{
  C_ = C;
  n_ = C_.rows();
}

void
SDPSolverCPU::invertObjectiveSign()
{
  C_ = -1 * C_;
}

void
SDPSolverCPU::addEqualityConstraint(EigenSMatrix& Ai, double bi)
{
  m_++;
  A_.emplace_back(Ai);
  b_.emplace_back(bi);
}

void
SDPSolverCPU::addInequalityConstraint(EigenSMatrix& Gi, double hi)
{
  p_++;
  G_.emplace_back(Gi);
  h_.emplace_back(hi);
}

cuda_linalg::EigenDMatrix
SDPSolverCPU::solve()
{
  assert(A_.size() == m_);
  assert(G_.size() == p_);

	Model::t solver = new Model("SDP Solver"); auto _M = finally([&]() { solver->dispose(); });
  // "finally" <- This is like delete or free of C/C++
  // but I don't know why this has to be called first

  // Setting up the variables
  Variable::t X = solver->variable("X", Domain::inPSDCone(n_));

  // Make Mosek Objective Matrix
  auto objective_mosek = eigen2fusion(C_);

  // Objective
  solver->objective(ObjectiveSense::Minimize, Expr::dot(objective_mosek, X));

  // Equality Constraints
  for(int i = 0; i < m_; i++)
  {
    auto equality_constraint_mosek = eigen2fusion(A_[i]);
    std::string const_name = "EqConstraint" + std::to_string(i);
    solver->constraint(const_name.c_str(), 
                       Expr::dot(equality_constraint_mosek, X), 
                       Domain::equalsTo(b_[i]));
  }

  // Inequality Constraints
  for(int i = 0; i < p_; i++)
  {
    auto inequality_constraint_mosek = eigen2fusion(G_[i]);
    std::string const_name = "IneqConstraint" + std::to_string(i);
    solver->constraint(const_name.c_str(), 
                       Expr::dot(inequality_constraint_mosek, X), 
                       Domain::greaterThan(h_[i]));
  }

  solver->setSolverParam("numThreads", "8");
  solver->solve();

  // Get Objective value
  primal_obj_ = solver->primalObjValue();

  // Solution (x vector)
  //auto sol_x = *(X->slice( nint( {0, 2} ), nint( {1, n_} ))->level());
  //auto sol_y = *(X->slice( nint( {1, 2} ), nint( {2, n_} ))->level());

  cuda_linalg::EigenDMatrix dense_matrix(n_, n_);
  for(int i = 0; i < n_; i++)
  {
    auto one_row = *(X->slice( nint( {i, 0} ), nint( {i + 1, n_} ))->level());
    for(int j = 0; j < n_; j++)
      dense_matrix(i, j) = one_row[j];
  }

  return dense_matrix;
  // decompSVD(dense_matrix);
}

std::vector<std::vector<double>>
SDPSolverCPU::getResult() const
{
  return solution_;
}

double
SDPSolverCPU::getObjectiveValue() const
{
  return primal_obj_;
}

void
SDPSolverCPU::decompSVD(cuda_linalg::EigenDMatrix& matrix)
{
  constexpr int k_target_rank = 2;
  int n = matrix.rows();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(matrix);
  // eigen value is sorted in the increasing order, so we select m from tail
  Eigen::VectorXd eigenvalues  = es.eigenvalues().tail(k_target_rank).reverse();
  Eigen::MatrixXd eigenvectors = es.eigenvectors().rightCols(k_target_rank).rowwise().reverse();
  Eigen::MatrixXd R(k_target_rank, n);
  for (int i = 0; i < k_target_rank; ++i) 
    R.row(i) = std::sqrt(eigenvalues(i)) * eigenvectors.col(i).transpose();

  solution_.resize(2, std::vector<double>(n_ - 2));

  for(int i = 0; i < n - 2; i++)
  {
    solution_[0][i] = matrix(0, i + 2);
    solution_[1][i] = matrix(1, i + 2);
  }
}

}
