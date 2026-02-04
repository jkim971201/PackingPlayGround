#include <fstream>

#include "Util.h"
#include "MacroPlacer.h"
#include "TargetFunction.h"
#include "Painter.h"

#include "nonlinear_solver/AdamSolver.h"

namespace macroplacer
{

void 
MacroPlacer::refineMacroPlace()
{
  auto refine_start = getChronoNow();

  float x_min = static_cast<float>(coreLx_);
  float y_min = static_cast<float>(coreLy_);
  float x_max = static_cast<float>(coreUx_);
  float y_max = static_cast<float>(coreUy_);

  std::shared_ptr<TargetFunction> function
    = std::make_shared<TargetFunction>(
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
  printf("Refine          finished (takes %5.2f s)\n", refine_time);

  // for sweep experiments
  std::ofstream log_output;
  log_output.open("temp_log.txt");

  log_output << function->getHpwl() << " ";
  log_output << function->getSumOverlap() << std::endl;
  printf("REPORT %f %f\n", function->getHpwl(), function->getSumOverlap());
}

}
