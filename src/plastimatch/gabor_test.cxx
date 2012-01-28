/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>

#include "itk_gabor.h"
#include "itk_image.h"
#include "itk_image_load.h"
#include "print_and_exit.h"
#include "pstring.h"


int 
main (int argc, char *argv[])
{
    if (argc != 2) {
        printf ("Usage: gabor_test image\n");
        exit (-1);
    }
    FloatImageType::Pointer image = itk_image_load_float (argv[1], 0);

    itk_gabor (image);
    return 0;
}
