#pragma once

#include <filesystem>
#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QColor>
#include <QScreen>
#include <unordered_map>
#include <vector>

#include "Packer.h"

namespace myPacker
{

class Rect;

class Painter : public QWidget
{
	public:

		// w : window width
		// h : window height
		// color : background color
		Painter(QSize size, QColor color, int optW, int optH);

		// Setters
		void setRectFillColor(QColor color) { rectFillColor_ = color; }
		void setRectLineColor(QColor color) { rectLineColor_ = color; }

		void setQRect(std::vector<Rect*>& rects);

	protected:

		void paintEvent(QPaintEvent* event);
		// This function is not for users

	private:

		void init();

		QSize  windowSize_;
		QColor baseColor_;
		QColor rectFillColor_;
		QColor rectLineColor_;

		std::vector<QRect> rectVector_;

		void drawRect(QPainter* painter, QRect& rect, QColor rectColor = Qt::white, QColor rectLineColor_ = Qt::black);
		void drawRect(QPainter* painter, int lx, int ly, int w, int h);

		int w_;
		int h_;
		int optW_;
		int optH_;
		int offset_;
};

}
