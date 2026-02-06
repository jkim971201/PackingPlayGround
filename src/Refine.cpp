#include <fstream>

#include "Util.h"
#include "MacroPlacer.h"
#include "Painter.h"

#include "problem_instance/RefineFloorplan.h"
#include "problem_instance/RefineQCQP.h"
#include "nonlinear_solver/AdamSolver.h"

namespace macroplacer
{

void 
MacroPlacer::refineQCQP()
{
  auto refine_start = getChronoNow();

  float scale_x = 1.0f / static_cast<float>(coreUx_ - coreLx_);
  float scale_y = 1.0f / static_cast<float>(coreUy_ - coreLy_);

  float offset_x = (coreLx_ + coreUx_) / 2.0f;
  float offset_y = (coreLy_ + coreUy_) / 2.0f;

  std::shared_ptr<RefineQCQP> augmented_lagrangian
    = std::make_shared<RefineQCQP>(
      scale_x,  /* scale_x  */
      scale_y,  /* scale_y  */
      offset_x, /* offset_x */
      offset_y, /* offset_y */
      -0.5f,    /* x_min */
      -0.5f,    /* y_min */
      +0.5f,    /* x_max */
      +0.5f,    /* y_max */
      painter_,
      movable_,
      Lmm_,
      Lmf_xf_,
      Lmf_yf_,
      ineq_constraint_);

  std::unique_ptr<AdamSolver> adam_solver
    = std::make_unique<AdamSolver>(augmented_lagrangian);

  const int max_phase = 10;
  for(int phase = 0; phase < max_phase; phase++)
  {
    adam_solver->solve();
    augmented_lagrangian->updateSlack();
    augmented_lagrangian->updateDual();
  }

  const double refine_time = evalTime(refine_start);
  printf("refineQCQP      finished (takes %5.2f s)\n", refine_time);
}

void 
MacroPlacer::refineRectangular()
{
  auto refine_start = getChronoNow();

  float x_min = static_cast<float>(coreLx_);
  float y_min = static_cast<float>(coreLy_);
  float x_max = static_cast<float>(coreUx_);
  float y_max = static_cast<float>(coreUy_);

  std::shared_ptr<RefineFloorplan> function
    = std::make_shared<RefineFloorplan>(
      x_min, y_min, x_max, y_max,
      painter_,
      movable_, net_ptrs_, pin_ptrs_);

  std::unique_ptr<AdamSolver> adam_solver
    = std::make_unique<AdamSolver>(function);

  // Restart ADAM
  const int max_phase = 200;
  for(int phase = 0; phase < max_phase; phase++)
  {
    float scale = static_cast<float>(phase + 1) / static_cast<float>(max_phase);
    function->scaleArea(scale);
    adam_solver->solve();
    //painter_->saveImage(phase, function->getHpwl(), function->getSumOverlap());
  }

  const double refine_time = evalTime(refine_start);
  printf("refineFloorplan finished (takes %5.2f s)\n", refine_time);

  // for sweep experiments
  std::ofstream log_output;
  log_output.open("temp_log.txt");

  log_output << function->getHpwl() << " ";
  log_output << function->getSumOverlap() << std::endl;
  printf("REPORT %f %f\n", function->getHpwl(), function->getSumOverlap());
}

}
