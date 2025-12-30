#include "Net.h"

#include <limits>

namespace macroplacer
{

Net::Net(int id) : id_(id), wl_(0) {}

int 
Net::id() const 
{ 
  return id_; 
}

int 
Net::wl() const 
{ 
  return wl_; 
} 

std::vector<Pin>&
Net::getPins()
{
  return pins_;
}

const std::vector<Pin>&
Net::getPins() const
{
  return pins_;
}

void 
Net::addPin(Macro* macro_ptr)
{
  pins_.emplace_back(macro_ptr, this);
}

void
Net::updateWL()
{
  int max_cx = 0;
  int max_cy = 0;
  int min_cx = std::numeric_limits<int>::max();
  int min_cy = std::numeric_limits<int>::max();

  for(const auto& pin : pins_)
  {
    max_cx = std::max(pin.getCx(), max_cx);
    max_cy = std::max(pin.getCy(), max_cy);
    min_cx = std::min(pin.getCx(), min_cx);
    min_cy = std::min(pin.getCy(), min_cy);
  }

  wl_ = max_cx - min_cx + max_cy - min_cy;
}

}
