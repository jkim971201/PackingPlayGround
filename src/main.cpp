#include <iostream>

#include "MacroPlacer.h"

using namespace macroplacer;

int    cmd_argc;
char** cmd_argv;

char* get_cmd_option(char ** begin, char ** end, const std::string & option)
{
  char** itr = std::find(begin, end, option);
  if(itr != end && ++itr != end) return *itr;
  else return 0;
}

bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
  return std::find(begin, end, option) != end;
}

int main(int argc, char **argv)
{
  if(argc < 4)
  {
    std::cout << "Need 3 input files (.blocks .pl .nets)\n";
    exit(0);
  }

  std::string block_file_str;
  std::string net_file_str;
  std::string pl_file_str;

  if(cmd_option_exists(argv, argv + argc, "-block") )
    block_file_str = std::string( get_cmd_option(argv, argv + argc, "-block") );
  else
  {
    printf("Please give .block file\n");
    exit(0);
  }

  if(cmd_option_exists(argv, argv + argc, "-net") )
    net_file_str = std::string( get_cmd_option(argv, argv + argc, "-net") );
  else
  {
    printf("Please give .net file\n");
    exit(0);
  }

  if(cmd_option_exists(argv, argv + argc, "-pl") )
    pl_file_str = std::string( get_cmd_option(argv, argv + argc, "-pl") );
  else
  {
    printf("Please give .pl file\n");
    exit(0);
  }

  MacroPlacer mpl;

  std::filesystem::path block_file(block_file_str);
  std::filesystem::path pl_file(pl_file_str);
  std::filesystem::path nets_file(net_file_str);

  mpl.readFile(block_file, pl_file, nets_file);
  //mpl.prepareVisualization();

  mpl.run();

  //mpl.show();

  return 0;
}
