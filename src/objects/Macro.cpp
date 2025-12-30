#include "Macro.h"
#include "Pin.h"

namespace macroplacer
{

Macro::Macro(const std::string& name, int lx, int ly, int w, int h)
  : lx_       (lx   ),
    ly_       (ly   ),
    w_        (w    ),
    h_        (h    ),
    name_     (name ),
    isPacked_ (false)
{}

int Macro::lx() const { return lx_; }
int Macro::ly() const { return ly_; }

int Macro::w() const { return w_; }
int Macro::h() const { return h_; }

bool Macro::isPacked() const { return isPacked_; }
std::string_view Macro::name() const { return name_; }

void Macro::setPacked() { isPacked_ = true; }  
void Macro::addPin(Pin* pin) { pins_.push_back(pin); }

void
Macro::move(int dx, int dy)
{
  lx_ += dx;
  ly_ += dy;

  for(auto& pin : pins_)
  {
    pin->setCx(pin->getCx() + dx);
    pin->setCy(pin->getCy() + dy);
  }
}

void
Macro::setLx(int newLx)
{
  int dx = newLx - lx_;
  lx_ = newLx;
  for(auto& pin : pins_)
    pin->setCx(pin->getCx() + dx);
}

void
Macro::setLy(int newLy)
{
  int dy = newLy - ly_;
  ly_ = newLy;
  for(auto& pin : pins_)
    pin->setCy(pin->getCy() + dy);
}

}
