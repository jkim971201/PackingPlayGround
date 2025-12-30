#include <iostream>

#include "MacroPlacer.h"

using namespace macroplacer;

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cout << "No input file. Please give .macros file...\n";
    exit(0);
  }

  MacroPlacer mpl;

  std::filesystem::path txtfile = argv[1];

  mpl.readFile(txtfile);

  //////////////////////////////////////////////////
  // Make your own macro placement algorithm!!!
  //pack.naivePacking();
  //////////////////////////////////////////////////

  mpl.show(argc, argv);

  return 0;
}
