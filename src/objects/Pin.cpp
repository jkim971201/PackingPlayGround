#include "Pin.h"
#include "Net.h"
#include "Macro.h"

namespace macroplacer
{

Pin::Pin(Macro* macro, Net* net)
  : macro_(macro), net_(net)
{
  cx_ = macro->getCx();
  cy_ = macro->getCy();
}

Net*
Pin::getNet() 
{
  return net_;
}

Macro* 
Pin::getMacro() 
{ 
  return macro_; 
}

const Net*
Pin::getNet() const
{
  return net_;
}

const Macro* 
Pin::getMacro() const
{ 
  return macro_; 
}

std::string_view
Pin::getMacroName() const
{ 
  return macro_->getName();
}

int 
Pin::getCx() const 
{ 
  return cx_; 
}

int 
Pin::getCy() const 
{ 
  return cy_; 
}

void 
Pin::setCx(int new_cx) 
{ 
  cx_ = new_cx; 
}

void 
Pin::setCy(int new_cy) 
{ 
  cy_ = new_cy; 
}

}
