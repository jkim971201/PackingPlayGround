#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>

#include "Painter.h"

namespace myPacker
{

Painter::Painter(QSize size, QColor color, int optW, int optH)
	: windowSize_ (size ),
		baseColor_  (color),
		w_          (    0),
		h_          (    0),
		optW_       ( optW),
		optH_       ( optH),
		offset_     (   20)
{
	init();
}

void
Painter::init()
{
	w_ = windowSize_.width()  / 4;
	h_ = windowSize_.height() / 4;

	printf("Width  of the window is : %d\n", w_);
	printf("Height of the window is : %d\n", h_);

	this->resize(w_, h_);

  QPalette palette( Painter::palette() );
  palette.setColor( backgroundRole(), baseColor_ );

  this->setPalette(palette);

  rectFillColor_ = Qt::red;
  rectLineColor_ = Qt::black;

	this->setWindowTitle( "Rectangle Packing Visualization" );
}

void
Painter::drawRect(QPainter* painter, QRect& rect, QColor rectColor, QColor lineColor)
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
Painter::setQRect(std::vector<Rect*>& rects)
{
	rectVector_.reserve(rects.size());

	for(auto& r : rects)
	{
		rectVector_.push_back( QRect( r->lx() + offset_, 
					                        h_ - 1 * offset_ - r->ly(), 
																	r->w(), 
																	-r->h() ) );
	}
}

void
Painter::paintEvent(QPaintEvent* event)
{
  //std::cout << "Start PaintEvent" << std::endl;

  QPainter painter(this);

  painter.setRenderHint(QPainter::Antialiasing);

	painter.setPen( QPen(Qt::white , 3) );

	QRect canvas(offset_, offset_, w_ - 2 * offset_, h_ - 2 * offset_);
	painter.drawRect( canvas );

	srand(time(NULL));
	for(auto& rect : rectVector_)
	{
		int r = rand() % 256;
		int g = rand() % 256;
		int b = rand() % 256;

    //std::cout << i << " -> " << r << " " << g << " " << b << std::endl;

		QColor color(r, g, b);

		drawRect( &painter, rect , color, Qt::black);
	}

	QRect optBoundary(offset_, h_ - offset_, optW_, -optH_);
	painter.setPen( QPen(Qt::red , 3) );
	painter.setBrush( QBrush(Qt::NoBrush) );
	painter.drawRect( optBoundary );

  //std::cout << "End PaintEvent" << std::endl;
}

}
