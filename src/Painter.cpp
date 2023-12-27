#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>

#include "Painter.h"

namespace myPacker
{

Painter::Painter(QSize size, QColor color, int coreUx, int coreUy, int coreLx, int coreLy, int64_t wl)
  : windowSize_ (            size),
    baseColor_  (           color),
    w_          (               0),
    h_          (               0),
    coreLx_     (          coreLx),
    coreLy_     (          coreLy),
    coreDx_     ( coreUx - coreLx),
    coreDy_     ( coreUy - coreLy),
    offset_     (             100),
    scaleX_     (             1.0),
    scaleY_     (             1.0),
    wl_         (              wl)
{
  init();
}

void
Painter::init()
{
  w_ = windowSize_.width()  / 4;
  h_ = windowSize_.height() / 4;

  scaleX_ = static_cast<float>(w_) / static_cast<float>(coreDx_);
  scaleY_ = static_cast<float>(h_) / static_cast<float>(coreDy_);

  printf("Width  of the window is : %d\n", w_);
  printf("Height of the window is : %d\n", h_);

  printf("ScaleX : %f\n", scaleX_);
  printf("ScaleY : %f\n", scaleY_);

  this->resize(w_ + 2 * offset_, h_ + 2 * offset_);

  QPalette palette( Painter::palette() );
  palette.setColor( backgroundRole(), baseColor_ );

  this->setPalette(palette);

  rectFillColor_ = Qt::red;
  rectLineColor_ = Qt::black;

  this->setWindowTitle( "Macro Placement Visualization" );
}

void
Painter::drawRect(QPainter* painter, QRectF& rect, QColor rectColor, QColor lineColor)
{
  painter->setPen( QPen(lineColor , 2) );
  painter->setBrush(rectColor);

  painter->drawRect( rect );
  painter->fillRect( rect , painter->brush() );
}

void
Painter::drawRect(QPainter* painter, int lx, int ly, int w, int h)
{
  QRect test(lx, ly, w, h);

  painter->drawRect( test );
  painter->fillRect( test, painter->brush() );
}

void
Painter::setQRect(std::vector<Macro*>& macros)
{
  rectVector_.reserve(macros.size());

  for(auto& macro : macros)
  {
    rectVector_.push_back( QRectF( (          macro->lx() - coreLx_) * scaleX_ + offset_, 
                                   (coreDy_ - macro->ly() + coreLy_) * scaleY_ + offset_, 
                                   +macro->w() * scaleX_, 
                                   -macro->h() * scaleY_) );
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
  QPointF p1((          net->pin1()->cx() - coreLx_) * scaleX_ + offset_, 
             (coreDy_ - net->pin1()->cy() + coreLy_) * scaleY_ + offset_);

  QPointF p2((          net->pin2()->cx() - coreLx_) * scaleX_ + offset_, 
             (coreDy_ - net->pin2()->cy() + coreLy_) * scaleY_ + offset_);

  painter->drawLine(p1, p2);
}

void
Painter::paintEvent(QPaintEvent* event)
{
  //std::cout << "Start PaintEvent" << std::endl;

  QPainter painter(this);

  painter.setRenderHint(QPainter::Antialiasing);

  painter.setPen( QPen(Qt::white , 3) );

  //printf("coreDx_ : %d scaleX_ : %f coreDx_ * scaleX_ : %f\n", coreDx_, scaleX_, coreDx_ * scaleX_);
  //printf("coreDy_ : %d scaleY_ : %f coreDy_ * scaleY_ : %f\n", coreDy_, scaleY_, coreDy_ * scaleY_);
  //QRectF core(offset_, optH_ * scaleY_ + offset_, optW_ * scaleX_, -optH_ * scaleY_);
  QRectF core(offset_, offset_, w_, h_);
  painter.setPen( QPen(Qt::blue , 2) );
  painter.setBrush( QBrush(Qt::NoBrush) );
  painter.drawRect( core );

  for(auto& rect : rectVector_)
    drawRect( &painter, rect , Qt::red, Qt::black);

  painter.setPen( QPen(Qt::green, 1) );
  for(auto& net : netVector_)
    drawNet( &painter, net );

  std::string wlInfo = "Total WL : " + std::to_string(wl_);

  painter.setPen( QPen(Qt::black, 1) );
  QFont font = painter.font();
  font.setPointSize(25);
  painter.setFont(font);
  painter.drawText(offset_, offset_ - 20, QString::fromStdString(wlInfo) );
  //std::cout << "End PaintEvent" << std::endl;
}

}
