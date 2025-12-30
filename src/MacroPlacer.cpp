#include <cstdio>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <map>

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

MacroPlacer::MacroPlacer()
  : painter_  (nullptr),
    coreLx_   (0),
    coreLy_   (0),
    coreUx_   (0),
    coreUy_   (0),
    totalWL_  (0)
{}

void
MacroPlacer::readBlock(const std::filesystem::path& block_file)
{
  printf("Read .blocks file : %s\n", block_file.string().c_str());
  std::string filename = block_file.string();
  size_t dot_pos = filename.find_last_of('.');
  std::string suffix = filename.substr(dot_pos + 1); 

  if(suffix != "blocks")
  {
    std::cerr << "This file is not .blocks file..." << std::endl;
    exit(1);
  }

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
      macro_insts_.emplace_back(macro_name, 0, 0, w, h, false);
    }
    else if(tokens.size() == 2)
    {
      std::string terminal_name = tokens[0];
      assert(tokens[1] == "terminal");

      // lx ly is unknown in the .blocks file
      macro_insts_.emplace_back(terminal_name, 0, 0, 0, 0, true);
    }
  }

  for(auto& inst : macro_insts_)
  {
    macro_ptrs_.push_back(&inst);
    std::string name(inst.getName());
    name_to_macro_ptr_[name] = &inst;
  }

  assert(macro_insts_.size() == num_hard_rect + num_terminals);
  printf("NumSoft : %d\n", num_soft_rect);
  printf("NumHard : %d\n", num_hard_rect);
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
    if(tokens[0].front() == '#')
      continue;

    if(tokens[0] == "UCSC")
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
    if(tokens[0].front() == '#')
      continue;

    if(tokens[0] == "UCLA")
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

  for(auto& net_inst : net_insts_)
  {
    net_ptrs_.push_back(&net_inst);
    for(auto& pin : net_inst.getPins())
      pin_ptrs_.push_back(&pin);
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

  printf("=====================================\n");
  printf("DB Info\n");
  printf("-------------------------------------\n");
  printf("NumMacro: %d\n", macro_ptrs_.size());
  printf("NumNet:   %d\n",   net_ptrs_.size());
  printf("NumPin:   %d\n",   pin_ptrs_.size());
  printf("Core (%d, %d) - (%d, %d)\n", coreLx_, coreLy_, coreUx_, coreUy_);
  printf("Initial HPWL: %ld\n", totalWL_);
  printf("=====================================\n");
}

void
MacroPlacer::updateWL()
{
  totalWL_ = 0;
  for(auto& net : net_ptrs_)
  {
    net->updateWL();
    totalWL_ += static_cast<int64_t>( net->wl() );
  }
}

void
MacroPlacer::naivePacking()
{
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
