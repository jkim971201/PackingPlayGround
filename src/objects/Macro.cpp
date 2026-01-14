#include "Macro.h"
#include "Pin.h"

namespace macroplacer
{

Macro::Macro(const std::string& name, int lx, int ly, int w, int h, bool is_terminal)
  : lx_(lx),
    ly_(ly),
    w_ (w),
    h_ (h),
    name_(name),
    is_terminal_(is_terminal)
{}

int Macro::getLx() const { return lx_; }
int Macro::getLy() const { return ly_; }
int Macro::getUx() const { return lx_ + w_; }
int Macro::getUy() const { return ly_ + h_; }
int Macro::getCx() const { return lx_ + w_ / 2; }
int Macro::getCy() const { return ly_ + h_ / 2; }

int Macro::getWidth() const { return w_; }
int Macro::getHeight() const { return h_; }

int Macro::getArea() const { return w_ * h_; }

bool Macro::isTerminal() const { return is_terminal_; }

std::string_view Macro::getName() const { return name_; }

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
