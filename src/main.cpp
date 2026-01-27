#include <iostream>

#include "MacroPlacer.h"

using namespace macroplacer;

int    cmd_argc;
char** cmd_argv;

int main(int argc, char **argv)
{
  if(argc != 4)
  {
    std::cout << "Need 3 input files (.blocks .pl .nets)\n";
    exit(0);
  }

  cmd_argc = argc;
  cmd_argv = argv;

  MacroPlacer mpl;

  std::filesystem::path block_file = argv[1];
  std::filesystem::path pl_file = argv[2];
  std::filesystem::path nets_file = argv[3];

  mpl.readFile(block_file, pl_file, nets_file);

  mpl.run();

  mpl.show();

  return 0;
}
