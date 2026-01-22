#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numbers>
#include <limits>
#include <map>

#include "SDPInstance.h"
#include "SDPSolverCPU.h"
#include "SDPSolverGPU.h"
#include "MacroPlacer.h"
#include "Painter.h"
#include "Util.h"

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

  //std::cout << "ObjMatrix" << std::endl;
  //std::cout << obj_matrix_dense << std::endl;
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
  obj_matrix.resize(num_movable + 2, num_movable + 2);

  obj_matrix.coeffRef(0, 0) = 1;
  obj_matrix.coeffRef(1, 1) = 1;
  for(int i = 0; i < Lmm.outerSize(); i++)
  {
    for(EigenSMatrix::InnerIterator it(Lmm, i); it; ++it)
      obj_matrix.coeffRef(it.row() + 2, it.col() + 2) = it.value();
  }

  for(int i = 0; i < num_movable; i++)
  {
    obj_matrix.coeffRef(i + 2, 0) = Lmf_xf(i);
    obj_matrix.coeffRef(i + 2, 1) = Lmf_yf(i);
    obj_matrix.coeffRef(0, i + 2) = Lmf_xf(i);
    obj_matrix.coeffRef(1, i + 2) = Lmf_yf(i);
  }

  // prune values that are smaller than ref_nonzero * epsilon
  obj_matrix.prune(/* ref_nonzero */ 1.0, /* epsilon */ 1e-3);

  sdp_inst->setObjectiveMatrix(obj_matrix);

  /* Equality Constraints */
  EigenSMatrix constr00;
  constr00.resize(num_movable + 2, num_movable + 2);
  constr00.coeffRef(0, 0) = 1;

  EigenSMatrix constr01;
  constr01.resize(num_movable + 2, num_movable + 2);
  constr01.coeffRef(0, 1) = 1;

  EigenSMatrix constr10;
  constr10.resize(num_movable + 2, num_movable + 2);
  constr10.coeffRef(1, 0) = 1;

  EigenSMatrix constr11;
  constr11.resize(num_movable + 2, num_movable + 2);
  constr11.coeffRef(1, 1) = 1;

  sdp_inst->addEqualityConstraint(constr00, 1);
  sdp_inst->addEqualityConstraint(constr01, 0);
  sdp_inst->addEqualityConstraint(constr10, 0);
  sdp_inst->addEqualityConstraint(constr11, 1);

  /* Non-overlap Constraints */
  int count = 0;
  for(int i = 0; i < num_movable; i++)
  {
    for(int j = i + 1; j < num_movable; j++)
    {
      EigenSMatrix constr_matrix;
      constr_matrix.resize(num_movable + 2, num_movable + 2);
      constr_matrix.coeffRef(i + 2, i + 2) = +1;
      constr_matrix.coeffRef(i + 2, j + 2) = -1;
      constr_matrix.coeffRef(j + 2, i + 2) = -1;
      constr_matrix.coeffRef(j + 2, j + 2) = +1;
      sdp_inst->addInequalityConstraint(constr_matrix, ineq_constraint(count));
      count++;
    }
  }

  return sdp_inst;
}

MacroPlacer::MacroPlacer()
  : painter_ (nullptr),
    coreLx_  (0),
    coreLy_  (0),
    coreUx_  (0),
    coreUy_  (0),
    totalWL_ (0)
{}

std::pair<double, double>
MacroPlacer::originalToScaled(double x, double y) const
{
  double scaled_x = (x - coreLx_) / (coreUx_ - coreLx_) * 0.5;
  double scaled_y = (y - coreLy_) / (coreUy_ - coreLy_) * 0.5;
  return {scaled_x, scaled_y};
}

std::pair<double, double>
MacroPlacer::scaledToOriginal(double x, double y) const
{
  double original_x = 2.0 * x * (coreUx_ - coreLx_) + coreLx_;
  double original_y = 2.0 * y * (coreUy_ - coreLy_) + coreLy_;
  return {original_x, original_y};
}

void
MacroPlacer::updateWL()
{
  totalWL_ = 0;
  for(const auto& net : net_ptrs_)
  {
    net->updateWL();
    totalWL_ += static_cast<int64_t>( net->wl() );
  }
}

void
MacroPlacer::run()
{
  auto laplacian_start = getChronoNow();

  computeFixedInfo();

  computeIneqConstraint(ineq_constraint_);

  createClusterLaplacian(L_);

  extractPartialLaplacian(xf_, yf_, L_, Lff_, Lmm_, Lmf_xf_, Lmf_yf_);

  const double laplacian_time = evalTime(laplacian_start);
  printf("createLaplacian finished (takes %5.2f s)\n", laplacian_time);

  auto sdp_start = getChronoNow();

  auto solution = solveSDP(Lmm_, Lmf_xf_, Lmf_yf_, ineq_constraint_);

  const double sdp_time = evalTime(sdp_start);
  printf("solveSDP        finished (takes %5.2f s)\n", sdp_time);

  int movable_id = 0;
  for(auto& macro : movable_)
  {
    auto [new_cx, new_cy] 
      = scaledToOriginal(solution[0][movable_id], solution[1][movable_id]);

    int dx = static_cast<int>(new_cx) - macro->getCx();
    int dy = static_cast<int>(new_cy) - macro->getCy();
    macro->move(dx, dy);
    movable_id++;
  }

  refineMacroPlace();
  printf("Refine          finished\n");

  updateWL();

  printf("TotalWL : %ld\n", totalWL_);

  writeBookshelf();
}

void
MacroPlacer::computeFixedInfo()
{
  int num_fixed = fixed_.size();
  xf_.resize(num_fixed);
  yf_.resize(num_fixed);

  int fixed_id = 0;
  for(const auto& macro : fixed_)
  {
    auto [scaled_x, scaled_y] = originalToScaled(macro->getCx(), macro->getCy());

    xf_(fixed_id) = scaled_x;
    yf_(fixed_id) = scaled_y;

    //printf("(%d, %d) -> (%f, %f)\n", macro->getCx(), macro->getCy(), scaled_x, scaled_y);
    fixed_id++;
  }
}

void
MacroPlacer::computeIneqConstraint(EigenVector& ineq_constraint)
{
  int num_movable = movable_.size();
  int num_overlap_pair = num_movable * (num_movable - 1) / 2;

  constexpr double k_pi = std::numbers::pi;

  ineq_constraint.resize(num_overlap_pair);

  const double scale = 1.0 / (coreUx_ - coreLx_) / (coreUy_ - coreLy_);

  int count = 0;
  for(int i = 0; i < num_movable; i++)
  {
    double rect_area_i = static_cast<double>(movable_[i]->getArea()) * scale;
    double radius_i = std::sqrt(rect_area_i / k_pi);
    for(int j = i + 1; j < num_movable; j++)
    {
      double rect_area_j = static_cast<double>(movable_[j]->getArea()) * scale;
      double radius_j = std::sqrt(rect_area_j / k_pi);
      double radius_sum = radius_i + radius_j;
      double radius_sum_square = radius_sum * radius_sum;
      ineq_constraint[count] = radius_sum_square;
      count++;
      //printf("Constraint: %d - %d Value: %f\n", i, j, radius_sum_square);
    }
  }

  assert(count == num_overlap_pair);
};

void
MacroPlacer::createClusterLaplacian(EigenSMatrix& L)
{
  std::unordered_map<const Macro*, int> macro_to_vertex_id;
  int vertex_id = 0;

  for(auto macro : movable_)
    macro_to_vertex_id[macro] = vertex_id++;
  for(auto macro : fixed_)
    macro_to_vertex_id[macro] = vertex_id++;

  // vertex_id - verteix_id - edge_weight
  std::vector<EigenTriplet> triplet_vector;

  const int num_total_vertex = movable_.size() + fixed_.size();
  // M2M : Movable to Movable
  // M2F : Movable to Fixed
  constexpr double k_weight_for_M2M = 1.0;
  constexpr double k_weight_for_M2F = 1.0;
  constexpr int max_degree = 25;

  for(const auto& net : net_ptrs_)
  {
    const auto& net_pins = net->getPins();
    const int net_degree = net_pins.size();
    // Since we are ignoring too large nets,
    // there can be some clusters that are not 
    // connected to any other vertex.
    if(net_degree < 2 || net_degree > max_degree)
      continue;

    for(int p1 = 0; p1 < net_degree - 1; p1++)
    {
      const Pin& pin1 = net_pins[p1];
      const Macro* pin1_macro = pin1.getMacro();

      auto find_vertex_id = macro_to_vertex_id.find(pin1_macro);
      assert(find_vertex_id != macro_to_vertex_id.end());
      int v1 = find_vertex_id->second;

      bool v1_fixed = false;
      if(pin1_macro->isTerminal() == true)
        v1_fixed = true;
      
      for(int p2 = p1 + 1; p2 < net_degree; p2++)
      {
        const Pin& pin2 = net_pins[p2];
        const Macro* pin2_macro = pin2.getMacro();

        auto find_vertex_id = macro_to_vertex_id.find(pin2_macro);
        assert(find_vertex_id != macro_to_vertex_id.end());
        int v2 = find_vertex_id->second;

        bool v2_fixed = false;
        if(pin2_macro->isTerminal() == true)
          v2_fixed = true;
                
        if(v1 == v2)
          continue;

        double weight = 1.0;
        if(v1_fixed == true || v2_fixed == true)
          weight = k_weight_for_M2F;
        else
          weight = k_weight_for_M2M;

        // L = D - A
        // For Diagonal Matrix
        triplet_vector.push_back(EigenTriplet(v1, v1, +weight));
        triplet_vector.push_back(EigenTriplet(v2, v2, +weight));

        // For Adjacency Matrix
        triplet_vector.push_back(EigenTriplet(v1, v2, -weight));
        triplet_vector.push_back(EigenTriplet(v2, v1, -weight));
      }
    }
  }

  L.resize(num_total_vertex, num_total_vertex);
  L.setFromTriplets(triplet_vector.begin(), triplet_vector.end());
}

void
MacroPlacer::extractPartialLaplacian(
  const EigenVector&  xf,
  const EigenVector&  yf,
  const EigenSMatrix& L, // Full Laplacian
        EigenSMatrix& Lff,
        EigenSMatrix& Lmm,
        EigenVector&  Lmf_xf,
        EigenVector&  Lmf_yf)
{
  const int num_movable = movable_.size();
  const int num_fixed   = fixed_.size();

  EigenSMatrix Lmf 
    = L.block(0,           // Start Index of Row (Y)
              num_movable, // Start Index of Col (X)
              num_movable, // Size of Y (numRow)
              num_fixed);  // Size of X (numCol)

  Lmm = L.block(0, 0, num_movable, num_movable);
  Lff = L.block(num_movable, num_movable, num_fixed, num_fixed);

  Lmf_xf.resize(num_movable);
  Lmf_yf.resize(num_movable);

  Lmf_xf = Lmf * xf;
  Lmf_yf = Lmf * yf;
}

std::vector<std::vector<double>>
MacroPlacer::solveSDP(
  const EigenSMatrix& Lmm, 
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf,
  const EigenVector&  ineq_constraint)
{
  int num_movable = movable_.size();

  std::vector<std::vector<double>> x_and_y;
  x_and_y.resize(2, std::vector<double>(num_movable));

  auto sdp_inst 
    = makeSDPInstance(num_movable, Lmm, Lmf_xf, Lmf_yf, ineq_constraint);

//  sdp_solver::SDPSolverCPU solver(sdp_inst);
//  EigenDMatrix solution = solver.solve();
//  for(int i = 0; i < solution.rows() - 2; i++)
//  {
//    x_and_y[0][i] = solution(0, i + 2);
//    x_and_y[1][i] = solution(1, i + 2);
//  }
//
//  printf("ObjVal : %f\n", solver.getObjectiveValue());

  checkCondition(sdp_inst->obj_matrix);

  // Test on GPU
  sdp_solver::SDPSolverGPU solver_gpu(sdp_inst);
  EigenDMatrix gpu_sol = solver_gpu.solve();

  //std::cout << "Solution from GPU" << std::endl;
  //std::cout << gpu_sol << std::endl;

  for(int i = 0; i < gpu_sol.rows() - 2; i++)
  {
    x_and_y[0][i] = gpu_sol(0, i + 2);
    x_and_y[1][i] = gpu_sol(1, i + 2);
  }

  return x_and_y;
}

int
MacroPlacer::show(int& argc, char* argv[])
{
  QApplication app(argc, argv);
  QSize size = app.screens()[0]->size();
  painter_ = std::make_unique<Painter>(size, Qt::darkGray, coreUx_, coreUy_, coreLx_, coreLy_, totalWL_);
  painter_->setQRect( macro_ptrs_ );
  painter_->setNetlist( net_ptrs_ );
  painter_->show();
  return app.exec();
}

}
