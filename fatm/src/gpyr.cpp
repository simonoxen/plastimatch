#include <assert.h>
#include <vector>
#include "mex.h"
#include "image.h"

/* Simulataneously shrink an image and its mask.  */
template <class T>
static void WeightedShrink
 (Image <GrayScale, T> *out_img,
  Image <GrayScale, T> *out_mask,
  const Image <GrayScale, T> &in_img,
  const Image <GrayScale, T> &in_mask)
{
  assert (in_img.Dims() == in_mask.Dims());
  /* GCS -- Why restrict to even sized imgs? */
  assert (in_img.Dims()(0) % 2 == 0 && in_img.Dims()(1) % 2 == 0);

  ImageDims in_dims = in_img.Dims(), out_dims = in_dims / 2;

  *out_img = Image <GrayScale, T> (out_dims);
  *out_mask = Image <GrayScale, T> (out_dims);

  const T *ii = in_img.Data(), *iw = in_mask.Data();
  T *oi = out_img->Data(), *ow = out_mask->Data();

  int y;
  for (y = 0; y < out_dims(1); y++)
    {
      int x;
      for (x = 0; x < out_dims(0); x++)
	{
	  T s = *ii * *iw +
	        *(ii + 1) * *(iw + 1) +
		*(ii + in_dims(0)) * *(iw + in_dims(0)) +
		*(ii + in_dims(0) + 1) * *(iw + in_dims(0) + 1);
	  T w = *iw +
	        *(iw + 1) +
		*(iw + in_dims(0)) +
		*(iw + in_dims(0) + 1);

	  if (w != 0)
	    *oi = s / w;
	  else
	    *oi = 0;
	  oi++;

	  *ow = w / 4;
	  ow++;

	  ii += 2;
	  iw += 2;
	}
      ii += in_dims(0);
      iw += in_dims(0);
    }
}

/* Build image pyramids for an image and its mask.  */
template <class T>
void BuildPyramid
 (std::vector<Image<GrayScale, T> > *img_pyr,
  std::vector<Image<GrayScale, T> > *mask_pyr,
  const Image<GrayScale, T> &img,
  const Image<GrayScale, T> &mask,
  int levels)
{
    Image<GrayScale, T> m = mask, i = img;
    img_pyr->push_back (i);
    mask_pyr->push_back (m);

    while (--levels > 0) {
	Image<GrayScale, T> ti (i), tm (m);
	WeightedShrink (&i, &m, ti, tm);
	img_pyr->push_back (i);
	mask_pyr->push_back (m);
    }
}

/* Instantiate the supported types.  */
template void BuildPyramid
 (std::vector<Image<GrayScale, double> > *img_pyr,
  std::vector<Image<GrayScale, double> > *mask_pyr,
  const Image<GrayScale, double> &img,
  const Image<GrayScale, double> &mask,
  int levels);
