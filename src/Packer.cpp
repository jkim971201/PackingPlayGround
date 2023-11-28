#include <stdio.h>
#include "Packer.h"

namespace myPacker
{

Packer::Packer()
{
	painter_ = nullptr;
}

void 
Packer::readFile(const std::filesystem::path& file)
{
  std::ifstream ifs;
	ifs.open(file);

  std::string line;

	int lx, ly, w, h = 0;

  std::getline(ifs, line);
	std::stringstream ss(line);

	ss >> boundW_;
	ss >> boundH_;

  while( !ifs.eof() )
  {
    std::getline(ifs, line);

    if(line.empty()) 
      continue;

    std::stringstream ss(line);
		ss >> lx;
		ss >> ly;
		ss >> w;
		ss >> h;

		rectInsts_.push_back( Rect(lx, ly, w, h) );
  }

	rectPtrs_.reserve(rectInsts_.size());
	for(auto& rect : rectInsts_)
		rectPtrs_.push_back( &rect );

	printf("Num Rect: %d\n", rectPtrs_.size());
	printf("Boundary (%d, %d)\n", boundW_, boundH_);
}

void
Packer::naivePacking()
{
	//std::sort( rectPtrs_.begin(), rectPtrs_.end(), sortByHeight() );
	std::sort( rectPtrs_.begin(), rectPtrs_.end(), [](Rect* a, Rect* b){ return a->h() > b->h(); } );

	int xPos = 0;
	int yPos = 0;
	int largestHeightThisRow = 0;

	for(auto& rect : rectPtrs_)
	{
		if( ( xPos + rect->w() ) > boundW_ )
		{
			yPos += largestHeightThisRow;
			xPos = 0;
			largestHeightThisRow = 0;
		}

		if( ( yPos + rect->h() ) > boundH_ )
		{
			printf("Naive packing failed\n");
			break;
		}

		rect->setLx(xPos);
		rect->setLy(yPos);

		xPos += rect->w();

		if( rect->h() > largestHeightThisRow )
			largestHeightThisRow = rect->h();

		rect->setPacked();
	}
}

int
Packer::show(int& argc, char* argv[])
{
	QApplication app(argc, argv);
	QSize size = app.screens()[0]->size();
	painter_ = std::make_unique<Painter>(size, Qt::darkGray, boundW_, boundH_);
	painter_->setQRect( rectPtrs_ );
	painter_->show();
	return app.exec();
}

}
