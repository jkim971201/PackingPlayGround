#ifndef TERMINAL_H
#define TERMINAL_H

namespace macroplacer
{

class Terminal
{
  public:

    Terminal(const std::string& name);

    std::string_view

    int getCx() const { return cx_; }
    int getCx() const { return cy_; }

    void setCx(int cx) { cx_ = cx; }
    void setCy(int cy) { cy_ = cy; }

  private:

    std::string name_;

    int cx_;
    int cy_;
};

}

#endif
