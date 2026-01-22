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

    Macro(const std::string& name, int lx, int ly, int w, int h, bool is_terminal);

    // Getters
    int getLx() const; 
    int getLy() const; 
    int getUx() const; 
    int getUy() const; 
    int getCx() const; 
    int getCy() const; 
    int getWidth() const; 
    int getHeight() const; 
    int getArea() const;
    bool isTerminal() const; 
    std::string_view getName() const; 

    std::vector<Pin*>& getPins();

    // NOTE
    // If you want to change the coordinates of macro location, 
    // then you also have to change the coordinates of its pins.
    // This will be done by setLx(), setLy(), move() automatically.
    
    // Setters
    void setLx(int lx);
    void setLy(int ly);
    void setCx(int cx);
    void setCy(int cy);
    void addPin(Pin* pin);
    void move(int dx, int dy);

  private:

    std::string name_;

    int  lx_;
    int  ly_;
    int  w_;
    int  h_;
    bool is_terminal_;

    std::vector<Pin*> pins_;
};

}

#endif
