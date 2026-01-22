#ifndef PIN_H
#define PIN_H

#include <string>

namespace macroplacer
{

class Net;
class Macro;

class Pin
{
  public:

    Pin(Macro* macro, Net* net);

    // Getters
    Net* getNet();
    Macro* getMacro();

    const Net* getNet() const;
    const Macro* getMacro() const;

    std::string_view getMacroName() const;

    int id() const;
    int getCx() const;
    int getCy() const;

    // Setters
    void setCx(int new_cx);
    void setCy(int new_cy);
    void setID(int id);

  private:

    int id_;
    int cx_;
    int cy_;
    Net* net_;
    Macro* macro_;
};

}

#endif
