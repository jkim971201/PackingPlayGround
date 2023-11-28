#include <iostream>
#include "Packer.h"

using namespace myPacker;

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		std::cout << "no input file\n";
		exit(0);
	}

	//QApplication app(argc, argv);
	//QSize size = app.screens()[0]->size();

	Packer pack;

	std::filesystem::path txtfile = argv[1];

	pack.readFile(txtfile);

	pack.naivePacking();

	pack.show(argc, argv);

	//return app.exec();
	
	return 0;
}
