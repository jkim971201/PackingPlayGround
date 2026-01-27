#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <numbers>

#include "Painter.h"
#include "objects/Pin.h"
#include "objects/Net.h"
#include "objects/Macro.h"

namespace macroplacer
{

Painter::Painter(
  QSize size, QColor color, int coreUx, int coreUy, int coreLx, int coreLy, int64_t wl)
  : coreLx_(         coreLx),
    coreLy_(         coreLy),
    coreDx_(coreUx - coreLx),
    coreDy_(coreUy - coreLy),
    scale_ (            1.0),
    wl_    (             wl)
{
  window_size_ = size;
  init();
}

void
Painter::init()
{
  path_to_save_image_ = "./plots";

  int window_w = window_size_.width()  * 3 / 5;
  int window_h = window_size_.height() * 3 / 5;
  window_length_ = std::min(window_w, window_h);
  offset_ = window_length_ * 0.05;

  float scale_x = static_cast<float>(window_length_) / static_cast<float>(coreDx_);
  float scale_y = static_cast<float>(window_length_) / static_cast<float>(coreDy_);
  scale_ = std::min(scale_x, scale_y);

  this->resize(window_length_ + 2 * offset_, window_length_ + 2 * offset_);

  QPalette palette( Painter::palette() );
  palette.setColor( backgroundRole(), Qt::black);

  this->setPalette(palette);
  this->setWindowTitle( "Macro Placement Visualization" );
}

void
Painter::drawRect(QPainter* painter, QRectF& rect, QColor rectColor, QColor lineColor)
{
  QPen pen_for_std;
  pen_for_std.setWidthF(3.5f);
  pen_for_std.setColor(lineColor);
  pen_for_std.setJoinStyle(Qt::PenJoinStyle::BevelJoin);
  pen_for_std.setStyle(Qt::PenStyle::SolidLine);

  QBrush brush_for_std(rectColor, Qt::BrushStyle::Dense6Pattern);
  painter->setBrush(brush_for_std);
  painter->setPen(pen_for_std);
  painter->drawRect( rect );
  painter->fillRect( rect , painter->brush() );
}

void
Painter::setMacros(std::vector<Macro*>& macros)
{
  macros_.reserve(macros.size());
  for(const auto& macro : macros)
    macros_.push_back(macro);
}

void
Painter::setNets(std::vector<Net*>& nets)
{
  nets_.reserve(nets.size());
  for(auto& net : nets)
    nets_.push_back(net);
}

void
Painter::drawNet(QPainter* painter, const Net* net)
{
  const auto pins = net->getPins();

  // com : center of mass
  int num_pin = pins.size();
  float com_x = 0;
  float com_y = 0;
  for(auto& pin : pins)
  {
    com_x += pin.getCx() / num_pin;
    com_y += pin.getCy() / num_pin;
  }

  for(auto pin : pins)
  {
    float pin_cx = pin.getCx();
    float pin_cy = pin.getCy();
    QPointF p1(pin_cx * scale_, pin_cy * scale_);
    QPointF p2(com_x * scale_, com_y * scale_);
    painter->drawLine(p1, p2);
  }
}

void
Painter::drawMacro(float k_scale, QPainter* painter, const Macro* macro)
{
  QPen pen_for_circle;
  pen_for_circle.setWidthF(3.5f);
  pen_for_circle.setColor(Qt::black);
  pen_for_circle.setJoinStyle(Qt::PenJoinStyle::BevelJoin);
  pen_for_circle.setStyle(Qt::PenStyle::SolidLine);

  QBrush brush_for_circle(Qt::gray, Qt::BrushStyle::Dense6Pattern);
  painter->setBrush(brush_for_circle);
  painter->setPen(pen_for_circle);

  int macro_w = macro->getWidth();
  int macro_h = macro->getHeight();
  double rect_area = macro_w * macro_h;

  constexpr double k_pi = std::numbers::pi;
  double radius = std::sqrt(rect_area / k_pi) * k_scale;

  double cx = macro->getCx() * k_scale;
  double cy = macro->getCy() * k_scale;

  //painter->drawEllipse(QPointF(cx, cy), radius, radius);

  float lx = macro->getLx() * k_scale;
  float ly = macro->getLy() * k_scale;
  float dx = macro->getWidth() * k_scale;
  float dy = macro->getHeight() * k_scale;
  QRectF macro_rect = QRectF(lx, ly, dx, dy);

  drawRect(painter, macro_rect, Qt::gray, Qt::black);

  if(macro->isTerminal() == false)
  {
    painter->save();
    QString name(macro->getName().data());
    QPen pen = painter->pen();
    QFont font = painter->font();
    //const qreal text_font_size = radius * 0.5;
    //font.setPointSizeF(text_font_size);
    painter->setFont(font);
    pen.setColor(Qt::white);
    painter->setPen(pen);
    painter->drawText(QPointF(cx, cy), name);
    painter->restore();
  }
}

void 
Painter::drawDieRect(float k_scale, QPainter* painter) const
{
  painter->save();
  float die_lx = coreLx_;
  float die_ly = coreLy_;
  float die_ux = coreLx_ + coreDx_;
  float die_uy = coreLy_ + coreDy_;

  float die_dx = coreDx_;
  float die_dy = coreDy_;

  // Draw Die
  QPen pen_for_die;
  pen_for_die.setColor(Qt::gray);
  pen_for_die.setStyle(Qt::PenStyle::DashDotLine);
  pen_for_die.setWidthF(4.0f);
  painter->setPen(pen_for_die);

  QRectF die_rect(k_scale * die_lx, k_scale * die_ly, k_scale * die_dx, k_scale * die_dy);
  painter->drawLine(die_rect.topLeft(), die_rect.topRight());
  painter->drawLine(die_rect.bottomLeft(), die_rect.bottomRight());
  painter->drawLine(die_rect.topLeft(), die_rect.bottomLeft());
  painter->drawLine(die_rect.topRight(), die_rect.bottomRight());
  painter->restore();
}

void
Painter::saveImage(int iter, float hpwl, float sum_overlap)
{
  float die_dx = static_cast<float>(coreDx_);
  float die_dy = static_cast<float>(coreDy_);

  const float k_scale_x = 500 / die_dx;
  const float k_scale_y = 500 / die_dy;
  const float k_scale = std::min(k_scale_x, k_scale_y);
  constexpr float k_margin = 0.2;

  int image_size_x = static_cast<int>(k_scale * (1 + k_margin) * die_dx);
  int image_size_y = static_cast<int>(k_scale * (1 + k_margin) * die_dy);

  QImage image(image_size_x, image_size_y, QImage::Format_RGB32);
  image.fill(Qt::black);
  QPainter painter(&image); 
  // When image is null (isNull() returns true),
  // QPainter will be not active.

  float offset_x = k_scale * k_margin * die_dx / 2.0;
  float offset_y = k_scale * k_margin * die_dy / 2.0;
  painter.translate(offset_x, offset_y);

  drawDieRect(k_scale, &painter);
  
  for(auto macro_ptr : macros_)
    drawMacro(k_scale, &painter, macro_ptr);

  auto get_padded_string = [] (size_t size, const std::string& str)
  {
    if(str.size() < size)
      return std::string(size - str.size(), ' ') + str;
    else
      return str;
  };

  std::string info;
  info += "Iter: ";
  info += get_padded_string(5, std::to_string(iter));
  info += "   HPWL: ";
  info += get_padded_string(13, std::to_string(static_cast<int64_t>(hpwl)) );
  info += "   SumOverlap: ";
  info += std::to_string(sum_overlap);

  QFont font = painter.font();
  font.setBold(true);
  const qreal text_font_size = k_scale * 0.025 * die_dy;
  font.setPointSizeF(text_font_size);
  painter.setFont(font);
  painter.scale(1.0, -1.0);
  auto pen_for_text = painter.pen();
  pen_for_text.setColor(Qt::white);
  painter.setPen(pen_for_text);

  QString metric_qstring(info.c_str());
  painter.drawText(0, offset_y / 2.0, metric_qstring);

  std::ostringstream iter_string;
  iter_string << std::setw(4) << std::setfill('0') << iter;

  std::string jpeg_name = iter_string.str() + ".jpeg";
  std::filesystem::path file_name = path_to_save_image_ / jpeg_name;
  QImage image_to_save = image.mirrored(false, true);
  image_to_save.save(file_name.string().c_str(), "jpeg", 50);
}

void
Painter::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen( QPen(Qt::white , 3) );

  painter.translate(offset_, offset_);

  drawDieRect(scale_, &painter);

  painter.setPen( QPen(Qt::darkGreen, 1) );
  for(const auto net : nets_)
    drawNet(&painter, net);

  for(const auto macro : macros_)
    drawMacro(scale_, &painter, macro);

  QFont font = painter.font();
  font.setBold(true);
  const qreal text_font_size = scale_ * 0.025 * coreDx_;
  font.setPointSizeF(text_font_size);
  painter.setFont(font);
  auto pen_for_text = painter.pen();
  pen_for_text.setColor(Qt::white);
  painter.setPen(pen_for_text);

  std::string wlInfo = "Total WL : " + std::to_string(wl_);
  painter.drawText(0, -offset_ * 0.2, QString::fromStdString(wlInfo) );
}

}
