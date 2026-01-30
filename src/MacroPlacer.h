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

#include "EigenDef.h"

class QApplication;

namespace macroplacer
{

using namespace cuda_linalg;

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

    void run();

    void prepareVisualization();

    int show();

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

    // Refine.cpp
    void refineMacroPlace();

    // MacroPlacer.cpp
    std::pair<double, double> originalToScaled(double x, double y) const;
    std::pair<double, double> scaledToOriginal(double x, double y) const;

    void updateWL();
    void computeFixedInfo();
    void createClusterLaplacian(EigenSMatrix& L);
    void extractPartialLaplacian(
      const EigenVector&  xf,
      const EigenVector&  yf,
      const EigenSMatrix& L, 
            EigenSMatrix& Lff,
            EigenSMatrix& Lmm,
            EigenVector&  Lmf_xf,
            EigenVector&  Lmf_yf);

    void computeIneqConstraint(EigenVector& ineq_constraint);

    // SDPRelaxation.cpp
    void suggestBySDPRelaxation(
      bool  use_gpu,
      const EigenSMatrix& Lmm,
      const EigenVector&  Lmf_xf,
      const EigenVector&  Lmf_yf,
      const EigenVector&  ineq_constraint);

    std::vector<std::vector<double>> solveSDP_CPU(
      const EigenSMatrix& Lmm,
      const EigenVector&  Lmf_xf,
      const EigenVector&  Lmf_yf,
      const EigenVector&  ineq_constraint);

    std::vector<std::vector<double>> solveSDP_GPU(
      const EigenSMatrix& Lmm,
      const EigenVector&  Lmf_xf,
      const EigenVector&  Lmf_yf,
      const EigenVector&  ineq_constraint);

    // Suggest.cpp
    void suggestByQP(
      const EigenSMatrix& Lmm,
      const EigenVector&  Lmf_xf,
      const EigenVector&  Lmf_yf);

    void suggestByRandomStart();

    // FileIO.cpp
    void readBlock(const std::filesystem::path& file);
    void readPlacement(const std::filesystem::path& file);
    void readNet(const std::filesystem::path& file);
    void initCore();
    void writeBookshelf() const;
    void writePl(std::string_view pl_file_name) const;
    void writeScl(std::string_view scl_file_name) const;
    void writeNodes(std::string_view nodes_file_name) const;
    void writeNets(std::string_view nets_file_name) const;

    // Members
    std::string design_name_;

    int coreLx_;
    int coreLy_;
    int coreUx_;
    int coreUy_;
    int num_terminals_;

    int64_t totalWL_;

    std::vector<Net>   net_insts_;
    std::vector<Net*>  net_ptrs_;
    
    std::vector<Pin*>  pin_ptrs_;

    std::vector<Macro>  macro_insts_;
    std::vector<Macro*> macro_ptrs_;

    std::vector<Macro*> movable_;
    std::vector<Macro*> fixed_;

    std::unique_ptr<QApplication> qapp_;
    std::shared_ptr<Painter> painter_;

    std::unordered_map<std::string, Macro*> name_to_macro_ptr_;

    // For SDP
    EigenVector  xm_;     // Movable x
    EigenVector  ym_;     // Movalbe y

    EigenVector  xf_;     // Fixed x
    EigenVector  yf_;     // Fixed y

    EigenVector  Lmf_xf_; // Lmf * xf
    EigenVector  Lmf_yf_; // Lmf * yf
    EigenVector  ineq_constraint_; // r_i^2 + r_j^2

    EigenSMatrix L_;      // Full Laplacian
    EigenSMatrix Lmm_;    // Laplacian between movable cells
    EigenSMatrix Lff_;    // Laplacian between fixed   cells
};

}

#endif
