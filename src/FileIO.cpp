#include <cstdio>
#include <iostream> // for std::cerr
#include <cassert>
#include <fstream>

#include "MacroPlacer.h"
#include "Painter.h"
#include "objects/Pin.h"
#include "objects/Net.h"
#include "objects/Macro.h"

namespace macroplacer
{

inline std::vector<std::string>
tokenize(std::string_view line, std::string_view dels)
{
  std::string token;
  std::vector<std::string> tokens;

  for(auto itr = line.begin(); itr < line.end(); itr++)
  {
    bool is_del = dels.find(*itr) != std::string_view::npos;
    if(is_del || std::isspace(*itr))
    {
      if(!token.empty())
      {
        tokens.push_back(std::move(token));
        token.clear();
      }
    }
    else
      token.push_back(*itr);
  }

  if(!token.empty()) 
    tokens.push_back(std::move(token));
  return tokens;
}

void
MacroPlacer::readBlock(const std::filesystem::path& block_file)
{
  printf("Read .blocks file : %s\n", block_file.string().c_str());
  std::string filename = block_file.string();
  size_t dot_pos = filename.find_last_of('.');
  size_t slash_pos   = filename.find_last_of('/');
  std::string suffix = filename.substr(dot_pos + 1); 

  if(suffix != "blocks")
  {
    std::cerr << "This file is not .blocks file..." << std::endl;
    exit(1);
  }

  design_name_ = filename.substr(slash_pos + 1, dot_pos - slash_pos - 1);

  std::ifstream ifs;
  ifs.open(block_file);

  std::string line;

  int num_soft_rect = 0;
  int num_hard_rect = 0;
  int num_terminals = 0;

  while(std::getline(ifs, line))
  {
    if(line.empty() == true)
      continue;

    auto tokens = tokenize(line, "(),");
    if(tokens.size() <= 1)
      continue;

    if(tokens[0].front() == '#')
      continue;

    if(tokens[0] == "NumSoftRectangularBlocks")
    {
      num_soft_rect = std::stoi(tokens[2]);
      assert(num_soft_rect == 0);
    }
    else if(tokens[0] == "NumHardRectilinearBlocks")
      num_hard_rect = std::stoi(tokens[2]);
    else if(tokens[0] == "NumTerminals")
      num_terminals = std::stoi(tokens[2]);

    if(tokens.size() == 11)
    {
      std::string macro_name = tokens[0];
      std::string hard_rect = tokens[1];
      int num_point = std::stoi(tokens[2]);
      assert(hard_rect == "hardrectilinear");
      assert(num_point == 4);

      int x1 = std::stoi(tokens[3]);
      int y1 = std::stoi(tokens[4]);

      int x2 = std::stoi(tokens[5]);
      int y2 = std::stoi(tokens[6]);

      int x3 = std::stoi(tokens[7]);
      int y3 = std::stoi(tokens[8]);

      int x4 = std::stoi(tokens[9]);
      int y4 = std::stoi(tokens[10]);

      assert(x1 == x2);
      assert(x3 == x4);
      assert(y1 == y4);
      assert(y2 == y3);
      assert(x3 > x2);
      assert(y2 > y1);

      int w = x3 - x2;
      int h = y2 - y1;

      // lx ly is unknown in the .blocks file
      macro_insts_.emplace_back(macro_name, w, h, false);
    }
    else if(tokens.size() == 5)
    {
      std::string macro_name = tokens[0];
      std::string soft_rect = tokens[1];
      assert(soft_rect == "softrectangular");

      int area = std::stoi(tokens[2]);
      float min_aspect_ratio = std::stof(tokens[3]);
      float max_aspect_ratio = std::stof(tokens[4]);
      macro_insts_.emplace_back(macro_name, area, min_aspect_ratio, max_aspect_ratio);
    }
    else if(tokens.size() == 2)
    {
      std::string terminal_name = tokens[0];
      assert(tokens[1] == "terminal");

      // lx ly is unknown in the .blocks file
      macro_insts_.emplace_back(terminal_name, 0, 0, true);
    }
  }

  for(auto& inst : macro_insts_)
  {
    macro_ptrs_.push_back(&inst);
    std::string name(inst.getName());
    name_to_macro_ptr_[name] = &inst;
  }

  assert(macro_insts_.size() == num_hard_rect + num_terminals);
  //printf("NumSoft : %d\n", num_soft_rect);
  //printf("NumHard : %d\n", num_hard_rect);

  num_terminals_ = num_terminals;
}

void
MacroPlacer::readPlacement(const std::filesystem::path& pl_file)
{
  printf("Read .pl     file : %s\n", pl_file.string().c_str());
  std::string filename = pl_file.string();
  size_t dot_pos = filename.find_last_of('.');
  std::string suffix = filename.substr(dot_pos + 1); 

  if(suffix != "pl")
  {
    std::cerr << "This file is not .pl file..." << std::endl;
    exit(1);
  }

  std::ifstream ifs;
  ifs.open(pl_file);
  std::string line;

  while(std::getline(ifs, line))
  {
    if(line.empty() == true)
      continue;

    auto tokens = tokenize(line, "(),");
    if(tokens.size() <= 1)
      continue;

    if(tokens[0].front() == '#')
      continue;

    if(tokens[0] == "UCSC" || tokens[0] == "UCLA")
      continue;
  
    assert(tokens.size() == 3);

    std::string macro_name = tokens[0];
    int lx = std::stoi(tokens[1]);
    int ly = std::stoi(tokens[2]);

    auto find_itr = name_to_macro_ptr_.find(macro_name);
    assert(find_itr != name_to_macro_ptr_.end());
    Macro* macro_ptr = find_itr->second;
    macro_ptr->setLx(lx);
    macro_ptr->setLy(ly);
    //printf("PL-> Macro: %s (%d, %d)\n", macro_name.c_str(), macro_ptr->getLx(), macro_ptr->getLy());
  }
}

void
MacroPlacer::readNet(const std::filesystem::path& net_file)
{
  printf("Read .nets   file : %s\n", net_file.string().c_str());
  std::string filename = net_file.string();
  size_t dot_pos = filename.find_last_of('.');
  std::string suffix = filename.substr(dot_pos + 1); 

  if(suffix != "nets")
  {
    std::cerr << "This file is not .nets file..." << std::endl;
    exit(1);
  }

  std::ifstream ifs;
  ifs.open(net_file);
  std::string line;

  int num_nets = 0;
  int num_pins = 0;

  int num_nets_read = 0;
  int num_pins_read = 0;

  auto read_one_net = [&] (int net_degree)
  {
    int num_pins_added = 0;
    int net_id = num_nets_read;

    Net new_net(net_id);

    while(num_pins_added < net_degree)
    {
      std::getline(ifs, line);
      if(line.empty() == true)
        continue;

      auto tokens = tokenize(line, "");
      if(tokens.size() <= 1)
        continue;

      if(tokens[0].front() == '#')
        continue;

      assert(tokens.size() == 2);
      std::string macro_name = tokens[0];
      std::string offset = tokens[1];

      auto find_itr = name_to_macro_ptr_.find(macro_name);
      assert(find_itr != name_to_macro_ptr_.end());
      Macro* macro_ptr = find_itr->second;
      new_net.addPin(macro_ptr);

      num_pins_read++;
      num_pins_added++;
    }

    net_insts_.push_back(new_net);
    assert(num_pins_added == net_degree);
    num_nets_read++;
  };

  while(std::getline(ifs, line))
  {
    if(line.empty() == true)
      continue;

    auto tokens = tokenize(line, "");
    if(tokens.size() <= 1)
      continue;

    if(tokens[0].front() == '#')
      continue;

    if(tokens[0] == "UCSC" || tokens[0] == "UCLA")
     continue;

    if(tokens[0] == "NumNets")
      num_nets = std::stoi(tokens[2]);

    if(tokens[0] == "NumPins")
      num_pins = std::stoi(tokens[2]);

    if(tokens[0] == "NetDegree")
    {
      int net_degree = std::stoi(tokens[2]);
      read_one_net(net_degree);
    }
  }

  assert(num_nets == net_insts_.size());
  assert(num_pins == num_pins_read);

  // NOTE: This order MUST be kept
  // net0 pin0
  //      pin1
  //      pin2
  // net1 pin3
  //      pin4
  //      pin5
  //      pin6
  //      ...
  int pin_id = 0;
  for(auto& net_inst : net_insts_)
  {
    net_ptrs_.push_back(&net_inst);
    for(auto& pin : net_inst.getPins())
    {
      pin.setID(pin_id++);
      pin_ptrs_.push_back(&pin);
      Macro* macro = pin.getMacro();
      macro->addPin(&pin);
    }
  }
}

void
MacroPlacer::initCore()
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = 0;
  int max_y = 0;

  float sum_area = 0.0f;
  for(const auto macro_ptr : macro_ptrs_)
  {
    int area = macro_ptr->getWidth() * macro_ptr->getHeight();
    sum_area += static_cast<float>(area);

    min_x = std::min(macro_ptr->getLx(), min_x);
    min_y = std::min(macro_ptr->getLy(), min_y);

    max_x = std::max(macro_ptr->getUx(), max_x);
    max_y = std::max(macro_ptr->getUy(), max_y);
  }

  coreLx_ = min_x;
  coreLy_ = min_y;
  coreUx_ = max_x;
  coreUy_ = max_x;
  //int core_area = (coreUx_ - coreLx_) * (coreUy_ - coreLy_);
  //printf("SumArea  : %d\n", static_cast<int>(sum_area));
  //printf("CoreArea : %d\n", core_area);
}

void 
MacroPlacer::readFile(
  const std::filesystem::path& block_file,
  const std::filesystem::path& pl_file,
  const std::filesystem::path& nets_file)
{
  readBlock(block_file);
  readPlacement(pl_file);
  readNet(nets_file);
  updateWL();

  initCore();

  for(auto macro_ptr : macro_ptrs_)
  {
    if(macro_ptr->isTerminal() == true)
      fixed_.push_back(macro_ptr);
    else 
      movable_.push_back(macro_ptr);
  }

  printf("=====================================\n");
  printf("DB Info\n");
  printf("-------------------------------------\n");
  printf("Design  : %s\n", design_name_.c_str());
  printf("NumNode : %d\n", macro_ptrs_.size());
  printf("NumTerm : %d\n", num_terminals_);
  printf("NumNet  : %d\n", net_ptrs_.size());
  printf("NumPin  : %d\n", pin_ptrs_.size());
  printf("Core (%d, %d) - (%d, %d)\n", coreLx_, coreLy_, coreUx_, coreUy_);
  printf("Initial HPWL: %ld\n", totalWL_);
  printf("=====================================\n");
}

void
MacroPlacer::writeBookshelf() const
{
  std::string output_design_name
   = design_name_ + "_bookshelf";

  std::string output_dir = output_design_name;

  std::string aux_file_name = output_design_name + ".aux";
  std::string pl_file_name = output_design_name + ".pl";
  std::string nets_file_name = output_design_name + ".nets";
  std::string scl_file_name = output_design_name + ".scl";
  std::string nodes_file_name = output_design_name + ".nodes";
  std::string wts_file_name = output_design_name + ".wts";

  std::string aux_file_name_dir = output_dir + "/" + aux_file_name;
  std::string pl_file_name_dir = output_dir + "/" +  pl_file_name;
  std::string nets_file_name_dir = output_dir + "/" + nets_file_name;
  std::string scl_file_name_dir = output_dir + "/" + scl_file_name;
  std::string nodes_file_name_dir = output_dir + "/" + nodes_file_name;
  std::string wts_file_name_dir = output_dir + "/" + wts_file_name;

  std::string command = "mkdir -p " + output_dir;
  std::system(command.c_str());

  // Step #1. Write .aux
  std::ofstream aux_output;
  aux_output.open(aux_file_name_dir);

  aux_output << "RowBasedPlacement :" << " ";
  aux_output << pl_file_name << " ";
  aux_output << nets_file_name << " ";
  aux_output << scl_file_name << " ";
  aux_output << nodes_file_name << " ";
  // If wts filename is not written in aux, 
  // ntuplace3 binaray makes segmentation fault.
  aux_output << wts_file_name;

  aux_output.close();
  printf("Write results to %s\n", aux_file_name.c_str());

  writePl(pl_file_name_dir);
  writeNodes(nodes_file_name_dir);
  writeScl(scl_file_name_dir);
  writeNets(nets_file_name_dir);
}

void
MacroPlacer::writePl(std::string_view pl_file_name) const
{
  std::ofstream pl_output;
  pl_output.open(pl_file_name.data());

  // Print Headline
  pl_output << "UCLA pl 1.0" << std::endl;
  pl_output << "# Created by SkyPlace (jkim97@postech.ac.kr)" << std::endl;
  pl_output << std::endl;

  for(const auto& macro : macro_insts_)
  {
    int lx_to_write = macro.getLx();
    int ly_to_write = macro.getLy();

    pl_output << macro.getName() << " ";
    pl_output << lx_to_write << " ";
    pl_output << ly_to_write << " : N";
    if(macro.isTerminal() == true)
      pl_output << " /FIXED";
    pl_output << std::endl;
  }
  
  pl_output.close();
  printf("Write results to %s\n", pl_file_name.data());
}

void
MacroPlacer::writeScl(std::string_view scl_file_name) const
{
  constexpr int k_row_h = 10;
  constexpr int k_sitewidth = 1;
  const int num_rows = (coreUy_ - coreLy_) / k_row_h;
  const int num_sites = (coreUx_ - coreLx_) / k_sitewidth;

  std::ofstream scl_output;
  scl_output.open(scl_file_name.data());

  // Print Headline
  scl_output << "UCLA scl 1.0" << std::endl;
  scl_output << "# Created by SkyPlace (jkim97@postech.ac.kr)" << std::endl;
  scl_output << std::endl;

  scl_output << "NumRows : " << num_rows << std::endl;
  scl_output << std::endl;

  int coordinate = coreLy_;
  for(int i = 0; i < num_rows; i++)
  {
    scl_output << "CoreRow Horizontal" << std::endl;
    scl_output << "  Coordinate   : "  << coordinate << std::endl;
    scl_output << "  Height       : "  << k_row_h << std::endl;
    scl_output << "  Sitewidth    : "  << k_sitewidth << std::endl;
    scl_output << "  Sitespacing  : 1" << std::endl;
    scl_output << "  Siteorient   : 1" << std::endl;
    scl_output << "  Sitesymmetry : 1" << std::endl;
    scl_output << "  SubrowOrigin : "  << coreLx_;
    scl_output << "  NumSites     : "  << num_sites << std::endl;
    scl_output << "End" << std::endl;
    coordinate += k_row_h;
  }

  scl_output.close();
  printf("Write results to %s\n", scl_file_name.data());
}

void
MacroPlacer::writeNodes(std::string_view nodes_file_name) const
{
  std::ofstream nodes_output;
  nodes_output.open(nodes_file_name.data());

  // Print Headline
  nodes_output << "UCLA nodes 1.0" << std::endl;
  nodes_output << "# Created by SkyPlace (jkim97@postech.ac.kr)" << std::endl;
  nodes_output << std::endl;

  nodes_output << "NumNodes : " << macro_insts_.size() << std::endl;
  nodes_output << "NumTerminals : " << num_terminals_ << std::endl;

  for(const auto& macro : macro_insts_)
  {
    int width  = macro.getWidth();
    int height = macro.getHeight();

    nodes_output << macro.getName() << " ";
    nodes_output << width << " ";
    nodes_output << height << " ";
    if(macro.isTerminal() == true)
      nodes_output << "terminal";
    nodes_output << std::endl;
  }
  
  nodes_output.close();
  printf("Write results to %s\n", nodes_file_name.data());
}

void
MacroPlacer::writeNets(std::string_view nets_file_name) const
{
  std::ofstream nets_output;
  nets_output.open(nets_file_name.data());

  // Print Headline
  nets_output << "UCLA nets 1.0" << std::endl;
  nets_output << "# Created by SkyPlace (jkim97@postech.ac.kr)" << std::endl;
  nets_output << std::endl;

  nets_output << "NumNets : " << net_insts_.size() << std::endl;
  nets_output << "NumPins : " << pin_ptrs_.size() << std::endl;
  nets_output << std::endl;

  int num_net_write = 0;
  for(const auto& net : net_insts_)
  {
    const auto pins = net.getPins();
    const int net_degree = pins.size();
    const std::string net_name = "net" + std::to_string(num_net_write++);

    nets_output << "NetDegree : " << net_degree << "  " << net_name << std::endl;
    for(const auto& pin : pins)
      nets_output << "  " << pin.getMacroName() << " I : 0 0" << std::endl;
  }
  
  nets_output.close();
  printf("Write results to %s\n", nets_file_name.data());
}

}
