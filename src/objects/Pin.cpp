#include "Pin.h"

namespace macroplacer
{

Pin::Pin(const std::string& name, int id, int cx, int cy, Macro* macro)
  : name_ (name),
    id_   (id), 
    cx_   (cx), 
    cy_   (cy), 
    net_  (nullptr), 
    macro_(macro)
{}

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

std::string_view 
Pin::getName() const 
{ 
  return name_; 
}

int 
Pin::getID() const 
{ 
  return id_;
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
Pin::setNet(Net* net) 
{
  net_ = net; 
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
