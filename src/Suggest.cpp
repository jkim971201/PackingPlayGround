#include <random>
#include <Eigen/IterativeLinearSolvers>

#include "Util.h"
#include "MacroPlacer.h"

namespace macroplacer
{

void
MacroPlacer::suggestByQP(
  const EigenSMatrix& Lmm,
  const EigenVector&  Lmf_xf,
  const EigenVector&  Lmf_yf)
{
  auto qp_start = getChronoNow();

  Eigen::BiCGSTAB<EigenSMatrix, Eigen::IdentityPreconditioner> solver;
  solver.setMaxIterations(10000);
  solver.compute(Lmm);

  EigenVector result_x = solver.solve(-Lmf_xf);
  double error_x = solver.error();

  EigenVector result_y = solver.solve(-Lmf_yf);
  double error_y = solver.error();

  const int num_movable = movable_.size();
  for(int i = 0; i < num_movable; i++)
  {
    double x_qp = result_x(i);
    double y_qp = result_y(i);
    auto [new_cx, new_cy] = scaledToOriginal(x_qp, y_qp);
    movable_[i]->setCx(new_cx);
    movable_[i]->setCy(new_cy);
  }

  const double qp_time = evalTime(qp_start);
  printf("solveQP         finished (takes %5.2f s)\n", qp_time);
}

void
MacroPlacer::suggestByRandomStart()
{
  // Place all movable cells in the center with Gaussian Noise.
  // See DREAMPlace TCAD`21 for details
  
  const double die_lx = static_cast<double>(coreLx_);
  const double die_ly = static_cast<double>(coreLy_);

  const double die_ux = static_cast<double>(coreUx_);
  const double die_uy = static_cast<double>(coreUy_);

  const double die_cx = (die_ux + die_lx) / 2.0;
  const double die_cy = (die_uy + die_ly) / 2.0;

  const double die_dx = die_ux - die_lx;
  const double die_dy = die_uy - die_ly;

  double mean_x = die_cx;  
  double mean_y = die_cy;  

  double random_init_dev_coeff_x = 0.05;
  double random_init_dev_coeff_y = 0.05;

  double deviation_x = die_dx * random_init_dev_coeff_x; // 0.05
  double deviation_y = die_dy * random_init_dev_coeff_y; // 0.05

  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(mean_x, deviation_x);
  std::normal_distribution<double> noise_y(mean_y, deviation_y);

  for(auto macro_ptr : movable_)
  {
    // For soft block, these will be zero.
    double width  = macro_ptr->getTempWidth();
    double height = macro_ptr->getTempHeight();
    double loc_x = noise_x(gen);
    double loc_y = noise_y(gen);

    loc_x = std::min(std::max(loc_x, die_lx + width  / 2), die_ux - width  / 2);
    loc_y = std::min(std::max(loc_y, die_ly + height / 2), die_uy - height / 2);

    macro_ptr->setCx(loc_x);
    macro_ptr->setCy(loc_y);
  }
}

}
