/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include "../maps-precomp.h" // Precomp header

#include <mrpt/slam/COccupancyGridMap2D.h>
#include <mrpt/system/os.h>
#include <mrpt/utils/round.h>

using namespace mrpt;
using namespace mrpt::slam;
using namespace mrpt::utils;
using namespace mrpt::poses;
using namespace std;



/*---------------------------------------------------------------
					getAsImage
  ---------------------------------------------------------------*/
void  COccupancyGridMap2D::getAsImage(
	utils::CImage	&img,
	bool verticalFlip,
	bool forceRGB,
	bool tricolor ) const
{
	if (!tricolor)
	{
		if (!forceRGB)
		{	// 8bit gray-scale
			img.resize(size_x,size_y,1,true); //verticalFlip);
			const cellType	*srcPtr = &map[0];
			unsigned char	*destPtr;
			for (unsigned int y=0;y<size_y;y++)
			{
				if (!verticalFlip)
						destPtr = img(0,size_y-1-y);
				else 	destPtr = img(0,y);
				for (unsigned int x=0;x<size_x;x++)
				{
					*destPtr++ = l2p_255(*srcPtr++);
				}
			}
		}
		else
		{	// 24bit RGB:
			img.resize(size_x,size_y,3,true); //verticalFlip);
			const cellType	*srcPtr = &map[0];
			unsigned char	*destPtr;
			for (unsigned int y=0;y<size_y;y++)
			{
				if (!verticalFlip)
						destPtr = img(0,size_y-1-y);
				else 	destPtr = img(0,y);
				for (unsigned int x=0;x<size_x;x++)
				{
					uint8_t c = l2p_255(*srcPtr++);
					*destPtr++ = c;
					*destPtr++ = c;
					*destPtr++ = c;
				}
			}
		}
	}
	else
	{
		// TRICOLOR: 0, 0.5, 1
		if (!forceRGB)
		{	// 8bit gray-scale
			img.resize(size_x,size_y,1,true); //verticalFlip);
			const cellType	*srcPtr = &map[0];
			unsigned char	*destPtr;
			for (unsigned int y=0;y<size_y;y++)
			{
				if (!verticalFlip)
						destPtr = img(0,size_y-1-y);
				else 	destPtr = img(0,y);
				for (unsigned int x=0;x<size_x;x++)
				{
					uint8_t c = l2p_255(*srcPtr++);
					if (c<120)
						c=0;
					else if (c>136)
						c=255;
					else c = 127;
					*destPtr++ = c;
				}
			}
		}
		else
		{	// 24bit RGB:
			img.resize(size_x,size_y,3,true); //verticalFlip);
			const cellType	*srcPtr = &map[0];
			unsigned char	*destPtr;
			for (unsigned int y=0;y<size_y;y++)
			{
				if (!verticalFlip)
						destPtr = img(0,size_y-1-y);
				else 	destPtr = img(0,y);
				for (unsigned int x=0;x<size_x;x++)
				{
					uint8_t c = l2p_255(*srcPtr++);
					if (c<120)
						c=0;
					else if (c>136)
						c=255;
					else c = 127;

					*destPtr++ = c;
					*destPtr++ = c;
					*destPtr++ = c;
				}
			}
		}
	}
}

/*---------------------------------------------------------------
					getAsImageFiltered
  ---------------------------------------------------------------*/
void  COccupancyGridMap2D::getAsImageFiltered(
	utils::CImage	&img,
	bool verticalFlip,
	bool forceRGB ) const
{
	getAsImage(img,verticalFlip,forceRGB);

	// Do filtering to improve the noisy peaks in grids:
	// ----------------------------------------------------
#if 0
	CTicTac  t;
#endif
	if (insertionOptions.CFD_features_gaussian_size!=0) 	img.filterGaussianInPlace( round( insertionOptions.CFD_features_gaussian_size ) );
	if (insertionOptions.CFD_features_median_size!=0) 		img.filterMedianInPlace( round( insertionOptions.CFD_features_median_size ) );
#if 0
	cout << "[COccupancyGridMap2D::getAsImageFiltered] Filtered in: " << t.Tac()*1000 << " ms" << endl;
#endif
}


