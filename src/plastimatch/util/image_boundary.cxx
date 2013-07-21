/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itkImage.h"

#include "image_boundary.h"
#include "itk_image_load.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "volume.h"

class Image_boundary_private {
public:
    Image_boundary_private () {
    }
public:
    UCharImageType::Pointer input_image;
    UCharImageType::Pointer output_image;
};

Image_boundary::Image_boundary ()
{
    d_ptr = new Image_boundary_private;
}

Image_boundary::~Image_boundary ()
{
    delete d_ptr;
}

void 
Image_boundary::set_input_image (const char* image_fn)
{
    d_ptr->input_image = itk_image_load_uchar (image_fn, 0);
}

void 
Image_boundary::set_input_image (
    const UCharImageType::Pointer image)
{
    d_ptr->input_image = image;
}

void 
Image_boundary::run ()
{
    /* Convert to Plm_image type */
    Plm_image pli_in (d_ptr->input_image);
    Volume *vol_in = pli_in.get_vol_uchar ();
    unsigned char *img_in = (unsigned char*) vol_in->img;

    /* Allocate output image */
    Plm_image *pli_out = pli_in.clone ();
    Volume *vol_out = pli_out->get_vol_uchar ();
    unsigned char *img_out = (unsigned char*) vol_out->img;

    /* Compute the boundary */
    for (plm_long k = 0, v = 0; k < vol_in->dim[2]; k++) {
        for (plm_long j = 0; j < vol_in->dim[1]; j++) {
            for (plm_long i = 0; i < vol_in->dim[0]; i++, v++)
            {
                /* Not a boundary unless one of the tests succeed */
                img_out[v] = 0;

                /* Only consider non-zero voxels */
                if (!img_in[v]) {
                    continue;
                }

                /* Non-zero edge pixels are boundary */
                if (k == 0 || k == vol_in->dim[2]-1
                    || j == 0 || j == vol_in->dim[1]-1
                    || i == 0 || i == vol_in->dim[0]-1)
                {
                    img_out[v] = 1;
                    continue;
                }

                /* Look for neighboring zero voxel in six-neighborhood */
                if (img_in[volume_index (vol_in->dim, i-1, j, k)] == 0) {
                    img_out[v] = 1;
                    continue;
                }
                if (img_in[volume_index (vol_in->dim, i+1, j, k)] == 0) {
                    img_out[v] = 1;
                    continue;
                }
                if (img_in[volume_index (vol_in->dim, i, j-1, k)] == 0) {
                    img_out[v] = 1;
                    continue;
                }
                if (img_in[volume_index (vol_in->dim, i, j+1, k)] == 0) {
                    img_out[v] = 1;
                    continue;
                }
                if (img_in[volume_index (vol_in->dim, i, j, k-1)] == 0) {
                    img_out[v] = 1;
                    continue;
                }
                if (img_in[volume_index (vol_in->dim, i, j, k+1)] == 0) {
                    img_out[v] = 1;
                    continue;
                }
            }
        }
    }

    /* Save the output image */
    d_ptr->output_image = pli_out->itk_uchar ();

    /* Clean up */
    delete pli_out;
}

UCharImageType::Pointer
Image_boundary::get_output_image ()
{
    return d_ptr->output_image;
}

UCharImageType::Pointer
do_image_boundary (UCharImageType::Pointer image)
{
    Image_boundary ib;
    ib.set_input_image (image);
    ib.run ();
    return ib.get_output_image ();
}
