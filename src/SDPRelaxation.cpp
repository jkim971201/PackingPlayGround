#include <iostream>

#include "Util.h"
#include "MacroPlacer.h"
#include "EigenMVN.h"

#include "sdp_solver/SDPInstance.h"
#include "sdp_solver/SDPSolverCPU.h"
#include "sdp_solver/SDPSolverGPU.h"

namespace macroplacer
{

void checkCondition(const EigenSMatrix& matrix_sparse)
{
  int num_rows = matrix_sparse.rows();
  EigenDMatrix matrix_dense(num_rows, num_rows);

  for(int i = 0; i < num_rows; i++)
	{
	  for(EigenSMatrix::InnerIterator it(matrix_sparse, i); it; ++it)
      matrix_dense(it.row(), it.col()) = static_cast<double>(it.value());
  }

  Eigen::SelfAdjointEigenSolver<EigenDMatrix> es(matrix_dense);

  double min_abs_ev = std::numeric_limits<double>::max();
  double max_abs_ev = std::numeric_limits<double>::min();

  int num_ev = es.eigenvalues().size();
  for(int i = 0; i < num_ev; i++)
  {
    double val = std::abs(es.eigenvalues()(i));
    min_abs_ev = std::min(val, min_abs_ev);
    max_abs_ev = std::max(val, max_abs_ev);
  }

  double cond = max_abs_ev / min_abs_ev;

  //std::cout << "EigenValues" << std::endl;
  //std::cout << es.eigenvalues() << std::endl;
  printf("Condition number of Objective Matrix : %f\n", cond);
}

std::shared_ptr<sdp_solver::SDPInstance> makeSDPInstance(
  int num_movable,
  const EigenSMatrix& Lmm,
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf,
  const EigenVector&  ineq_constraint)
{
  std::shared_ptr<sdp_solver::SDPInstance> sdp_inst
   = std::make_shared<sdp_solver::SDPInstance>();

  /* Objective Matrix */
  EigenSMatrix obj_matrix;
  obj_matrix.resize(2 * num_movable + 1, 2 * num_movable + 1);

  for(int i = 0; i < Lmm.outerSize(); i++)
  {
    for(EigenSMatrix::InnerIterator it(Lmm, i); it; ++it)
    {
      int row = it.row();
      int col = it.col();
      double val = it.value();
      obj_matrix.coeffRef(row + 1, col + 1) =  val;
      obj_matrix.coeffRef(row + 1 + num_movable, col + 1 + num_movable) = val;
    }
  }

  for(int i = 0; i < num_movable; i++)
  {
    obj_matrix.coeffRef(i + 1, 0) = Lmf_xf(i);
    obj_matrix.coeffRef(0, i + 1) = Lmf_xf(i);

    obj_matrix.coeffRef(i + 1 + num_movable, 0) = Lmf_yf(i);
    obj_matrix.coeffRef(0, i + 1 + num_movable) = Lmf_yf(i);
  }

  // prune values that are smaller than ref_nonzero * epsilon
  obj_matrix.prune(/* ref_nonzero */ 1.0, /* epsilon */ 1e-3);

  sdp_inst->setObjectiveMatrix(obj_matrix);

  /* Equality Constraints */
  EigenSMatrix constr00;
  constr00.resize(2 * num_movable + 1, 2 * num_movable + 1);
  constr00.coeffRef(0, 0) = 1;
  sdp_inst->addEqualityConstraint(constr00, 1);

  /* Non-overlap Constraints */
  int count = 0;
  for(int i = 0; i < num_movable; i++)
  {
    for(int j = i + 1; j < num_movable; j++)
    {
      EigenSMatrix constr_matrix;
      constr_matrix.resize(2 * num_movable + 1, 2 * num_movable + 1);

      int i_x = i + 1;
      int j_x = j + 1;
      constr_matrix.coeffRef(i_x, i_x) = +1;
      constr_matrix.coeffRef(i_x, j_x) = -1;
      constr_matrix.coeffRef(j_x, i_x) = -1;
      constr_matrix.coeffRef(j_x, j_x) = +1;

      int i_y = i + 1 + num_movable;
      int j_y = j + 1 + num_movable;
      constr_matrix.coeffRef(i_y, i_y) = +1;
      constr_matrix.coeffRef(i_y, j_y) = -1;
      constr_matrix.coeffRef(j_y, i_y) = -1;
      constr_matrix.coeffRef(j_y, j_y) = +1;

      sdp_inst->addInequalityConstraint(constr_matrix, ineq_constraint(count));
      count++;
    }
  }

  return sdp_inst;
}

void 
MacroPlacer::suggestBySDPRelaxation(
  bool  use_gpu,
  const EigenSMatrix& Lmm,
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf,
  const EigenVector&  ineq_constraint)
{
  auto sdp_start = getChronoNow();

  auto solution = (use_gpu == true) 
    ? solveSDP_GPU(Lmm, Lmf_xf, Lmf_yf, ineq_constraint)
    : solveSDP_CPU(Lmm, Lmf_xf, Lmf_yf, ineq_constraint);

  const double sdp_time = evalTime(sdp_start);
  printf("solveSDP        finished (takes %5.2f s)\n", sdp_time);

  int movable_id = 0;
  const int num_movable = movable_.size();
  for(auto& macro : movable_)
  {
    double x_sdp = solution[movable_id];
    double y_sdp = solution[movable_id + num_movable];
    auto [new_cx, new_cy] = scaledToOriginal(x_sdp, y_sdp);
    macro->setCx(new_cx);
    macro->setCy(new_cy);
    movable_id++;
  }

  refineQCQP();
}

std::vector<double>
MacroPlacer::solveSDP_CPU(
  const EigenSMatrix& Lmm, 
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf,
  const EigenVector&  ineq_constraint)
{
  int num_movable = movable_.size();

  std::vector<double> x_and_y(2 * num_movable);

  auto sdp_inst 
    = makeSDPInstance(num_movable, Lmm, Lmf_xf, Lmf_yf, ineq_constraint);

  sdp_solver::SDPSolverCPU solver(sdp_inst);
  EigenDMatrix solution = solver.solve();
  for(int i = 0; i < num_movable; i++)
  {
    x_and_y[i] = solution(0, i + 1);
    x_and_y[i + num_movable] = solution(0, i + 1 + num_movable);
  }

  return x_and_y;
}


std::vector<double>
MacroPlacer::solveSDP_GPU(
  const EigenSMatrix& Lmm, 
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf,
  const EigenVector&  ineq_constraint)
{
  int num_movable = movable_.size();

  std::vector<double> x_and_y(2 * num_movable);

  auto sdp_inst 
    = makeSDPInstance(num_movable, Lmm, Lmf_xf, Lmf_yf, ineq_constraint);

  //checkCondition(sdp_inst->obj_matrix);

  sdp_solver::SDPSolverGPU solver_gpu(sdp_inst);
  solver_gpu.setVerbose(false);
  EigenDMatrix gpu_sol = solver_gpu.solve();
  for(int i = 0; i < num_movable; i++)
  {
    x_and_y[i] = gpu_sol(0, i + 1);
    x_and_y[i + num_movable] = gpu_sol(0, i + 1 + num_movable);
  }

  //x_and_y = takeRandomization(gpu_sol);

  return x_and_y;
}

std::vector<double>
MacroPlacer::takeRandomization(const EigenDMatrix& sdp_sol)
{
  int num_movable = movable_.size();

  std::vector<double> randomized_solution(2 * num_movable);

  EigenVector mean(num_movable * 2);
  EigenDMatrix covar(num_movable * 2, num_movable  * 2);

  for(int i = 0; i < num_movable * 2; i++)
  {
    mean(i) = sdp_sol(0, i + 1);
    for(int j = 0; j < num_movable * 2; j++)
      covar(i, j) = sdp_sol(i + 1, j + 1);
  }

  covar = covar - mean * mean.transpose();

  Eigen::EigenMultivariateNormal<double> mv_norm(mean, covar);

  EigenDMatrix random_data = mv_norm.samples(1); 

  for(int i = 0; i < 2 * num_movable; i++)
    randomized_solution[i] = random_data(i, 0);

  return randomized_solution;
}

}
