#ifndef MACRO_PLACER_H
#define MACRO_PLACER_H

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "Painter.h"
#include "objects/Pin.h"
#include "objects/Net.h"
#include "objects/Macro.h"

namespace macroplacer
{

class Painter;

class MacroPlacer
{
  public:
    
    // Constructor
    MacroPlacer();

    // APIs
    void readFile(const std::filesystem::path& file);
    void naivePacking();

    int show(int& argc, char* argv[]);

    // Getters
    std::vector<Macro*>& macros() { return macroPtrs_; }
    std::vector<Net*>&     nets() { return   netPtrs_; }
    std::vector<Pin*>&     pins() { return   pinPtrs_; }

    int coreLx()  const { return coreLx_;  }
    int coreLy()  const { return coreLy_;  }
    int coreUx()  const { return coreUx_;  }
    int coreUy()  const { return coreUy_;  }
    int64_t totalWL() const { return totalWL_; }

  private:

    int coreLx_;
    int coreLy_;
    int coreUx_;
    int coreUy_;

    int64_t totalWL_;

    void updateWL();

    std::vector<Pin>   pinInsts_;
    std::vector<Pin*>  pinPtrs_;

    std::vector<Net>   netInsts_;
    std::vector<Net*>  netPtrs_;

    std::vector<Macro>  macroInsts_;
    std::vector<Macro*> macroPtrs_;

    std::unique_ptr<Painter> painter_;
};

}

#endif
