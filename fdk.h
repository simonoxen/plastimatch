/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_h_
#define _fdk_h_

typedef struct cb_image CB_Image;
struct cb_image
{
    int dim[2];         /* dim[0] = cols, dim[1] = rows */
    double ic[2];
    double matrix[12];
    double sad;
    double sid;
    double nrm[3];
    float* img;
};

#endif
