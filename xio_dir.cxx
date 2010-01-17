 /* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "bstrlib.h"

#include "plm_config.h"
#include "print_and_exit.h"
#include "xio_dir.h"

Xio_dir*
xio_dir_create (char *input_dir)
{
    Xio_dir *xd;
    xd = (Xio_dir*) malloc (sizeof (Xio_dir));
    strncpy (xd->path, input_dir, _MAX_PATH);
    return xd;
}

int
xio_dir_num_patients (Xio_dir* xd)
{
    itksys::Directory dir;
    if (!dir.Load (xd->path)) {
	printf ("Error\n");exit (-1);
    }
    return 0;
}

void
xio_dir_destroy (Xio_dir* xd)
{
}
