#include "Macro.h"
#include "Pin.h"

#include <cmath>

namespace macroplacer
{

// for HARD Block
Macro::Macro(const std::string& name, int w, int h, bool is_term)
  : lx_(0), ly_(0), w_(w), h_(h), name_(name), is_terminal_(is_term)
{
  area_ = w_ * h_;
  temp_w_ = w_;
  temp_h_ = h_;
  is_soft_block_ = false;
  min_ar_ = static_cast<float>(h_) / static_cast<float>(w_);
  max_ar_ = static_cast<float>(h_) / static_cast<float>(w_);
}

// for SOFT Block
Macro::Macro(const std::string& name, int area, float min_ar, float max_ar)
  : lx_(0), ly_(0), w_(0), h_(0), name_(name), area_(area), min_ar_(min_ar), max_ar_(max_ar)
{
  temp_w_ = 0.0f;
  temp_h_ = 0.0f;
  is_terminal_ = false;
  is_soft_block_ = true;
}

int Macro::getLx() const { return lx_; }
int Macro::getLy() const { return ly_; }
int Macro::getUx() const { return lx_ + w_; }
int Macro::getUy() const { return ly_ + h_; }
int Macro::getCx() const { return lx_ + w_ / 2; }
int Macro::getCy() const { return ly_ + h_ / 2; }

int Macro::getWidth() const { return w_; }
int Macro::getHeight() const { return h_; }
int Macro::getOriginalArea() const { return area_; }

float Macro::getMinAR() const { return min_ar_; }
float Macro::getMaxAR() const { return max_ar_; }

float Macro::getTempRatio() const { return temp_h_ / temp_w_; }
float Macro::getTempWidth() const { return temp_w_; }
float Macro::getTempHeight() const { return temp_h_; }
float Macro::getTempArea() const { return temp_w_ * temp_h_; }

bool Macro::isSoftBlock() const { return is_soft_block_; }
bool Macro::isTerminal() const { return is_terminal_; }

std::string_view Macro::getName() const { return name_; }

std::vector<Pin*>& Macro::getPins() { return pins_; }

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
Macro::setCx(int new_cx)
{
  lx_ = new_cx - w_ / 2;
  for(auto& pin : pins_)
    pin->setCx(new_cx);
}

void
Macro::setCy(int new_cy)
{
  ly_ = new_cy - h_ / 2;
  for(auto& pin : pins_)
    pin->setCy(new_cy);
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

void 
Macro::setTempShape(float scale, float aspect_ratio)
{
  float temp_area = static_cast<float>(area_) * scale;
  temp_w_ = std::sqrt(temp_area / aspect_ratio);
  temp_h_ = std::sqrt(temp_area * aspect_ratio);
}

}
