#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <map>

#include "SDPSolverCPU.h"
#include "MacroPlacer.h"
#include "Painter.h"
#include "Util.h"

namespace macroplacer
{

MacroPlacer::MacroPlacer()
  : painter_  (nullptr),
    coreLx_   (0),
    coreLy_   (0),
    coreUx_   (0),
    coreUy_   (0),
    totalWL_  (0)
{}

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

  createClusterLaplacian(L_);

  extractPartialLaplacian(xf_, yf_, L_, Lff_, Lmm_, Lmf_xf_, Lmf_yf_);

  const double laplacian_time = evalTime(laplacian_start);
  printf("createLaplacian finished (takes %5.2f s)\n", laplacian_time);

  auto sdp_start = getChronoNow();

  xm_ = solveSDP(Lmm_, Lmf_xf_);
  ym_ = solveSDP(Lmm_, Lmf_yf_);

  const double sdp_time = evalTime(sdp_start);
  printf("solveSDP        finished (takes %5.2f s)\n", sdp_time);

  int movable_id = 0;
  for(auto& macro : movable_)
  {
    int new_cx = xm_(movable_id);
    int new_cy = ym_(movable_id);
    int dx = new_cx - macro->getCx();
    int dy = new_cy - macro->getCy();
    macro->move(dx, dy);
    movable_id++;
  }

  updateWL();

  printf("TotalWL : %ld\n", totalWL_);
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
    xf_(fixed_id) = macro->getCx();
    yf_(fixed_id) = macro->getCy();
    fixed_id++;
  }
}

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

EigenVector 
MacroPlacer::solveSDP(
  const EigenSMatrix& Lmm, 
  const EigenVector& b)
{
  int N = Lmm.rows();

  EigenSMatrix M_0; // Objective
  EigenSMatrix M_3; // Constraint
  M_0.resize(N + 1, N + 1);
  M_3.resize(N + 1, N + 1);

  for(int i = 0; i < N; i++)
  {
    M_0.coeffRef(i + 1, i + 1) = Lmm.coeff(i, i);
    M_0.coeffRef(i + 1, 0) = b(i);
    M_0.coeffRef(0, i + 1) = b(i);
  }
  
  // M_3 has only one non-zero in (0, 0)
  M_3.coeffRef(0, 0) = 1;

  sdp_solver::SDPSolverCPU solver(N + 1);
  solver.setObjective(M_0);
  solver.addEqualityConstraint(M_3, 1);
  solver.solve();

  printf("ObjVal : %f\n", solver.getObjectiveValue());
  return solver.getResult();
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
