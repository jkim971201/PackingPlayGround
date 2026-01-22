#include "MacroPlacer.h"
#include "TargetFunction.h"

namespace macroplacer
{

void 
MacroPlacer::refineMacroPlace()
{
  std::shared_ptr<TargetFunction> function
    = std::make_shared<TargetFunction>(macro_ptrs_, net_ptrs_, pin_ptrs_);
}

}
