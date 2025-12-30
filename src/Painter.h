#ifndef PAINTER_H
#define PAINTER_H

#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QColor>
#include <QScreen>
#include <unordered_map>
#include <vector>

namespace macroplacer
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
    void setQRect    (std::vector<Macro*>& macros);
    void setNetlist  (std::vector<Net*>&     nets);

  protected:

    void paintEvent(QPaintEvent* event);
    // This function is not for users

  private:

    void init();

    QSize windowSize_;
    std::vector<QRectF> rectVector_;
    std::vector<const Net*> netVector_;

    void drawRect(QPainter* painter, QRectF& rect, QColor rectColor = Qt::white, QColor rectLineColor = Qt::black);
    void drawNet (QPainter* painter, const Net* net);

    int coreLx_;
    int coreLy_;
    int coreDx_;
    int coreDy_;
    int64_t wl_;

    float scale_;
    float offset_;
    float window_length_;
};

}

#endif
