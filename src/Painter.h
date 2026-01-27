#ifndef PAINTER_H
#define PAINTER_H

#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QColor>
#include <QScreen>

#include <vector>
#include <filesystem>
#include <unordered_map>

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
    Painter(
      QSize size, QColor color, int coreUx, int coreUy, int coreLx, int coreLy, int64_t wl);

    // Setters
    void setMacros(std::vector<Macro*>& macros);
    void setNets(std::vector<Net*>& nets);

    // APIs
    void saveImage(int iter, float hpwl, float sum_overlap);

  protected:

    void paintEvent(QPaintEvent* event);
    // This function is not for users

  private:

    void init();

    void drawRect(QPainter* painter, QRectF& rect, QColor rectColor = Qt::white, QColor rectLineColor = Qt::black);
    void drawNet(QPainter* painter, const Net* net);

    void drawDieRect(float k_scale, QPainter* painter) const;
    void drawMacro(float k_scale, QPainter* painter, const Macro* macro);

    QSize window_size_;

    int coreLx_;
    int coreLy_;
    int coreDx_;
    int coreDy_;
    int64_t wl_;

    float scale_;
    float offset_;
    float window_length_;

    std::filesystem::path path_to_save_image_;

    std::vector<Net*>   nets_;
    std::vector<Macro*> macros_;
};

}

#endif
