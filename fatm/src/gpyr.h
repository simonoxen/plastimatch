#ifndef GPYR_H
#define GPYR_H

#include <vector>
#include "image.h"

typedef std::vector<Image<GrayScale, double> > Gpyr;

template <class T>
void BuildPyramid
(std::vector<Image<GrayScale, T> > *img_pyr,
 std::vector<Image<GrayScale, T> > *mask_pyr,
 const Image<GrayScale, T> &img,
 const Image<GrayScale, T> &mask,
 int levels);

#endif
