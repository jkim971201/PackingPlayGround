#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include "Painter.h"

namespace myPacker
{

class Painter;

class Rect
{
	public:

		Rect(int lx, int ly, int w, int h)
			: lx_       (lx),
			  ly_       (ly),
				w_        ( w),
				h_        ( h),
				isPacked_ (false)
		{}

		// Getters
		int  lx()       const { return lx_;       }
		int  ly()       const { return ly_;       }
		int   w()       const { return w_;        }
		int   h()       const { return h_;        }
		bool isPacked() const { return isPacked_; }

		// Setters
		void setLx(int x) { lx_ = x; }
		void setLy(int y) { ly_ = y; }
		void setPacked()  { isPacked_ = true; }

	private:

		int  lx_;
		int  ly_;
		int  w_;
		int  h_;
		bool isPacked_;
};

class Packer
{
	public:
		
		// Constructor
		Packer();

		// APIs
		void readFile(const std::filesystem::path& file);
		void naivePacking();

		int show(int& argc, char* argv[]);

		// Getters
		std::vector<Rect*>& rects() { return rectPtrs_; }

	private:

		int boundW_;
		int boundH_;

		std::vector<Rect>  rectInsts_;
		std::vector<Rect*> rectPtrs_;
		std::unique_ptr<Painter> painter_;
};

}
