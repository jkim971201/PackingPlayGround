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

    // for HARD Block
    Macro(const std::string& name, int lx, int ly, int w, int h, bool is_terminal);

    // for SOFT Block
    Macro(const std::string& name, int lx, int ly, int w, int h, int area, bool is_terminal);

    // Getters
    int getLx() const; 
    int getLy() const; 
    int getUx() const; 
    int getUy() const; 
    int getCx() const; 
    int getCy() const; 
    int getWidth() const; 
    int getHeight() const; 
    int getOriginalArea() const;

    float getMinAR() const;
    float getMaxAR() const;

    float getTempWidth() const;
    float getTempHeight() const;
    float getTempArea() const;

    bool isSoftBlock() const;
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

    void setTempWidth(float val);
    void setTempHeight(float val);

  private:

    std::string name_;

    int  lx_;
    int  ly_;
    int  w_;
    int  h_;
    int  area_;

    float min_ar_;
    float max_ar_;

    float temp_w_;
    float temp_h_;

    bool is_soft_block_;
    bool is_terminal_;

    std::vector<Pin*> pins_;
};

}

#endif
