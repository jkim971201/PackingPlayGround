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

Painter::Painter(QSize size, QColor color, int coreUx, int coreUy, int coreLx, int coreLy, int64_t wl)
  : window_size_ (            size),
    coreLx_     (          coreLx),
    coreLy_     (          coreLy),
    coreDx_     ( coreUx - coreLx),
    coreDy_     ( coreUy - coreLy),
    scale_      (             1.0),
    wl_         (              wl)
{
  init();
}

void
Painter::init()
{
  int window_w = window_size_.width()  * 3 / 5;
  int window_h = window_size_.height() * 3 / 5;
  window_length_ = std::min(window_w, window_h);
  offset_ = window_length_ * 0.05;

  float scale_x = static_cast<float>(window_length_) / static_cast<float>(coreDx_);
  float scale_y = static_cast<float>(window_length_) / static_cast<float>(coreDy_);
  scale_ = std::min(scale_x, scale_y);

  printf("Width  of the window is : %d\n", window_w);
  printf("Height of the window is : %d\n", window_h);

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
Painter::drawCircle(QPainter* painter, const Macro* macro)
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
  double radius = std::sqrt(rect_area / k_pi) * scale_;

  double cx = macro->getCx() * scale_;
  double cy = macro->getCy() * scale_;

  painter->drawEllipse(QPointF(cx, cy), radius, radius);
  //painter->drawRect( rect );
  //painter->fillRect( rect , painter->brush() );

  if(macro->isTerminal() == false)
  {
    QString name(macro->getName().data());
    QPen pen = painter->pen();
    QFont font = painter->font();
    //const qreal text_font_size = radius * 0.5;
    //font.setPointSizeF(text_font_size);
    painter->setFont(font);
    pen.setColor(Qt::white);
    painter->setPen(pen);
    painter->drawText(QPointF(cx, cy), name);
  }
}

void
Painter::setQRect(std::vector<Macro*>& macros)
{
  rect_vector_.reserve(macros.size());
  for(const auto& macro : macros)
  {
    float lx = macro->getLx() * scale_;
    float ly = macro->getLy() * scale_;
    float dx = macro->getWidth() * scale_;
    float dy = macro->getHeight() * scale_;
    rect_vector_.push_back(QRectF(lx, ly, dx, dy));
    macro_vector_.push_back(macro);
  }
}

void
Painter::setNetlist(std::vector<Net*>& nets)
{
  for(auto& net : nets)
    netVector_.push_back(net);
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
Painter::paintEvent(QPaintEvent* event)
{
  QPainter painter(this);
  painter.setPen( QPen(Qt::white , 3) );

  painter.translate(offset_, offset_);

  float core_lx = coreLx_ * scale_;
  float core_ly = coreLy_ * scale_;
  float core_dx = coreDx_ * scale_;
  float core_dy = coreDy_ * scale_;
  QRectF core(core_lx, core_ly, core_dx, core_dy);

  QPen pen_for_die;
  pen_for_die.setColor(Qt::gray);
  pen_for_die.setStyle(Qt::PenStyle::DashDotLine);
  pen_for_die.setWidthF(4.0f);
  painter.setPen(pen_for_die);
  painter.setBrush( QBrush(Qt::NoBrush) );
  painter.drawRect( core );

  painter.setPen( QPen(Qt::darkGreen, 1) );
  for(auto& net : netVector_)
    drawNet( &painter, net );

  for(auto macro : macro_vector_)
    drawCircle( &painter, macro );

  //for(auto& rect : rect_vector_)
  //  drawRect( &painter, rect , Qt::gray, Qt::black);

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

  painter.scale(1.0, -1.0);
}

}
