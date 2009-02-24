/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_h_
#define _fdk_h_

typedef struct cb_image CB_Image;
struct cb_image
{
    int dim[2];         /* dim[0] = cols
			   dim[1] = rows */
    double ic[2];	/* Image Center
			   ic[0] = x
			   ic[1] = y     */
    double matrix[12];	// Projection matrix
    double sad;		// Distance: Source To Axis
    double sid;		// Distance: Source to Image
    double nrm[3];	// Ray from image center to source
    float* img;		// Pixel data
};

#endif
