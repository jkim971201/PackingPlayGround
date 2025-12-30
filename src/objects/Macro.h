#ifndef MACRO_H
#define MACRO_H

#include <vector>
#include <string>

namespace macroplacer
{

class Pin;

class Macro
{
  public:

    Macro(const std::string& name, int lx, int ly, int w, int h);

    // Getters
    int lx() const; 
    int ly() const; 
    int w() const; 
    int h() const; 
    bool isPacked() const; 
    std::string_view name() const; 

    // Setters
    // NOTE: If you want to change the coordinates of macro location, 
    //       then you also have to change the coordinates of its pins.
    //       This will be done by setLx(), setLy(), move() automatically.
    void setLx(int lx);
    void setLy(int ly);
    void setPacked();
    void addPin(Pin* pin);
    void move(int dx, int dy);

  private:

    std::string name_;

    int  lx_;
    int  ly_;
    int  w_;
    int  h_;
    bool isPacked_;

    std::vector<Pin*> pins_;
};

}

#endif
