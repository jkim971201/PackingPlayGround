#include <stdio.h>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <map>

#include "Packer.h"
#include "Painter.h"

namespace myPacker
{

void
Macro::move(int dx, int dy)
{
  lx_ += dx;
  ly_ += dy;

  for(auto& pin : pins_)
  {
    pin->setCx(pin->cx() + dx);
    pin->setCy(pin->cy() + dy);
  }
}

void
Macro::setLx(int newLx)
{
  int dx = newLx - lx_;
  lx_ = newLx;
  for(auto& pin : pins_)
    pin->setCx(pin->cx() + dx);
}

void
Macro::setLy(int newLy)
{
  int dy = newLy - ly_;
  ly_ = newLy;
  for(auto& pin : pins_)
    pin->setCy(pin->cy() + dy);
}

Packer::Packer()
  : painter_  (nullptr),
    coreLx_   (      0),
    coreLy_   (      0),
    coreUx_   (      0),
    coreUy_   (      0),
    totalWL_  (      0)
{}

void 
Packer::readFile(const std::filesystem::path& file)
{
  //std::cout << "Read " << file.string() << "..." << std::endl;
  
  std::string filename = file.string();
  size_t dotPos = filename.find_last_of('.');
  std::string suffix = filename.substr(dotPos + 1); 

  if(suffix != "macros")
  {
    std::cerr << "This file is not .macros file..." << std::endl;
    exit(1);
  }

  std::ifstream ifs;
  ifs.open(file);

  std::string line;
  std::string tempStr1;
  std::string tempStr2;

  int numMacro = 0;
  int numNet = 0;
  int numPin = 0;
  int lx, ly, w, h = 0;

  // READ DIE (CORE)
  std::getline(ifs, line);
  assert(line == "DIE");

  std::getline(ifs, line);
  std::stringstream dieInfo(line);

  dieInfo >> coreLx_;
  dieInfo >> coreLy_;
  dieInfo >> coreUx_;
  dieInfo >> coreUy_;
  
  // READ MACROS
  std::getline(ifs, line);
  std::stringstream macroInfo(line);

  macroInfo >> tempStr1;
  macroInfo >> tempStr2;

  assert(tempStr1 == "MACROS");
  numMacro = std::stoi(tempStr2);

  macroInsts_.reserve(numMacro);
  macroPtrs_.reserve(numMacro);

  std::map<std::string, Pin*> pinTable;

  while(1)
  {
    std::getline(ifs, line);
    //std::cout << "Line1: " << line << std::endl;

    if(line.empty()) 
      continue;

    if(line == "END MACROS")
      break;

    std::stringstream ss1(line);
    std::string macroName;
    ss1 >> macroName;

    //std::cout << "Macro Name: " << macroName << std::endl;
      
    std::getline(ifs, line);
    std::stringstream ss2(line);
    ss2 >> lx;
    ss2 >> ly;
    ss2 >> w;
    ss2 >> h;

    macroInsts_.push_back( Macro(macroName, lx, ly, w, h) );
    Macro* macroPtr = &(macroInsts_.back());
    macroPtrs_.push_back( macroPtr );

    std::getline(ifs, line);
    assert(line == "PINS");

    while(1)
    {
      std::getline(ifs, line);
      //std::cout << "Line2: " << line << std::endl;

      if(line.empty()) 
        continue;

      if(line == "END PINS")
        break;

      std::stringstream ss3(line);

      std::string pinName;
      int pinCx, pinCy = 0;
      ss3 >> pinName;
      ss3 >> pinCx;
      ss3 >> pinCy;

      //std::cout << "Pin Name : " << pinName << " Cx : " << pinCx << " Cy : " << pinCy << std::endl; 

      pinInsts_.push_back( Pin(pinName, numPin++, pinCx, pinCy, macroPtr) );
    }
  }

  for(auto& pin : pinInsts_)
  {
    Pin* pinPtr = &pin;
    pinPtrs_.push_back( pinPtr );
    pinPtr->macro()->addPin( pinPtr );
    pinTable[ pinPtr->name() ] = pinPtr;
    //std::cout << "Pin Name : " << pinPtr->name() << " MacroName from Pin   : " << pinPtr->macro()->name() << std::endl;
  }

  // READ NETS

  std::getline(ifs, line);
  std::stringstream netInfo(line);

  netInfo >> tempStr1;
  netInfo >> tempStr2;

  assert(tempStr1 == "NETS");
  numNet = std::stoi(tempStr2);

  netInsts_.reserve(numNet);
  netPtrs_.reserve(numNet);

  int numNetRead = 0;
  while(1)
  {
    std::getline(ifs, line);

    if(line.empty()) 
      continue;

    if(line == "END NETS")
      break;

    std::stringstream pinInfo(line);

    std::string pinName1, pinName2;

    pinInfo >> pinName1;
    pinInfo >> pinName2;

    Pin* pinPtr1;
    Pin* pinPtr2;

    if( pinTable.find(pinName1) != pinTable.end() )
      pinPtr1 = pinTable[pinName1];
    else
    {
      std::cout << pinName1 << " does not exist in the pinTable.\n";
      exit(0);
    }

    if( pinTable.find(pinName2) != pinTable.end() )
      pinPtr2 = pinTable[pinName2];
    else
    {
      std::cout << pinName2 << " does not exist in the pinTable.\n";
      exit(0);
    }

    netInsts_.push_back( Net(numNetRead++, pinPtr1, pinPtr2) );
    Net* netPtr = &(netInsts_.back());
    netPtrs_.push_back( netPtr );
    pinPtr1->setNet( netPtr );
    pinPtr2->setNet( netPtr );
  }

  updateWL();

  printf("=====================================\n");
  printf("DB Info\n");
  printf("-------------------------------------\n");
  printf("NumMacro: %d\n", macroPtrs_.size());
  printf("NumNet:   %d\n",   netPtrs_.size());
  printf("NumPin:   %d\n",   pinPtrs_.size());
  printf("Core (%d, %d) - (%d, %d)\n", coreLx_, coreLy_, coreUx_, coreUy_);
  printf("Initial HPWL: %ld\n", totalWL_);
  printf("=====================================\n");
}

void
Packer::updateWL()
{
  totalWL_ = 0;
  for(auto& net : netPtrs_)
  {
    net->updateWL();
    totalWL_ += static_cast<int64_t>( net->wl() );
  }
}

void
Packer::naivePacking()
{
  //std::sort( macroPtrs_.begin(), macroPtrs_.end(), sortByHeight() );
  std::sort( macroPtrs_.begin(), macroPtrs_.end(), [](Macro* a, Macro* b){ return a->h() > b->h(); } );

  int xPos = coreLx_;
  int yPos = coreLy_;
  int largestHeightThisRow = 0;

  for(auto& macro : macroPtrs_)
  {
    if( ( xPos - coreLx_ + macro->w() ) > coreUx_)
    {
      yPos += largestHeightThisRow;
      xPos = coreLx_;
      largestHeightThisRow = 0;
    }

    if( ( yPos - coreLy_ + macro->h() ) > coreUy_)
    {
      printf("Naive packing failed\n");
      break;
    }

    macro->setLx(xPos);
    macro->setLy(yPos);

    xPos += macro->w();

    if( macro->h() > largestHeightThisRow )
      largestHeightThisRow = macro->h();

    macro->setPacked();
  }

  updateWL();
  printf("WL: %ld\n", totalWL_);
}

int
Packer::show(int& argc, char* argv[])
{
  QApplication app(argc, argv);
  QSize size = app.screens()[0]->size();
  painter_ = std::make_unique<Painter>(size, Qt::darkGray, coreUx_, coreUy_, coreLx_, coreLy_, totalWL_);
  painter_->setQRect( macroPtrs_ );
  painter_->setNetlist( netPtrs_ );
  painter_->show();
  return app.exec();
}

}
