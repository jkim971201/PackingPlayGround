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
    void readFile(
      const std::filesystem::path& block_file,
      const std::filesystem::path& pl_file,
      const std::filesystem::path& nets_file);

    void naivePacking();

    int show(int& argc, char* argv[]);

    // Getters
    std::vector<Macro*>& macros() { return macro_ptrs_; }
    std::vector<Net*>&     nets() { return   net_ptrs_; }
    std::vector<Pin*>&     pins() { return   pin_ptrs_; }

    int coreLx()  const { return coreLx_;  }
    int coreLy()  const { return coreLy_;  }
    int coreUx()  const { return coreUx_;  }
    int coreUy()  const { return coreUy_;  }
    int64_t totalWL() const { return totalWL_; }

  private:

    void readBlock(const std::filesystem::path& file);
    void readPlacement(const std::filesystem::path& file);
    void readNet(const std::filesystem::path& file);
    void initCore();

    int coreLx_;
    int coreLy_;
    int coreUx_;
    int coreUy_;

    int64_t totalWL_;

    void updateWL();

    std::vector<Net>   net_insts_;
    std::vector<Net*>  net_ptrs_;
    
    std::vector<Pin*>  pin_ptrs_;

    std::vector<Macro>  macro_insts_;
    std::vector<Macro*> macro_ptrs_;

    std::unique_ptr<Painter> painter_;

    std::unordered_map<std::string, Macro*> name_to_macro_ptr_;
};

}

#endif
