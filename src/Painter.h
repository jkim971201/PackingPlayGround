#pragma once

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

class Macro;
class Net;

class Painter : public QWidget
{
  public:

    // w : window width
    // h : window height
    // color : background color
    Painter(QSize size, QColor color, int coreUx, int coreUy, int coreLx, int coreLy, int64_t wl);

    // Setters
    void setRectFillColor(QColor color) { rectFillColor_ = color; }
    void setRectLineColor(QColor color) { rectLineColor_ = color; }

    void setQRect    (std::vector<Macro*>& macros);
    void setNetlist  (std::vector<Net*>&     nets);
    void setWL       (int64_t wl) { wl_ = wl; }

  protected:

    void paintEvent(QPaintEvent* event);
    // This function is not for users

  private:

    void init();

    QSize  windowSize_;
    QColor baseColor_;
    QColor rectFillColor_;
    QColor rectLineColor_;

    std::vector<QRectF> rectVector_;
    std::vector<const Net*> netVector_;

    void drawRect(QPainter* painter, QRectF& rect, QColor rectColor = Qt::white, QColor rectLineColor = Qt::black);
    void drawRect(QPainter* painter, int lx, int ly, int w, int h);
    void drawNet (QPainter* painter, const Net* net);

    int w_;
    int h_;
    int coreLx_;
    int coreLy_;
    int coreDx_;
    int coreDy_;
    int offset_;

    int64_t wl_;

    float scaleX_;
    float scaleY_;
};

}
