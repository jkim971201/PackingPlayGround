#ifndef NET_H
#define NET_H

#include <vector>
#include "Pin.h"

namespace macroplacer
{

class Macro;

class Net
{
  public:

    Net(int id);

    int id() const;
    int wl() const; // NOTE: wl() has to be called after updateWL()
          
          std::vector<Pin>& getPins();
    const std::vector<Pin>& getPins() const;

    void updateWL();
    void addPin(Macro* macro_ptr);

  private:

    int id_;
    int wl_;
    std::vector<Pin> pins_;
};

}

#endif
