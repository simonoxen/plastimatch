/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itkTimeProbe.h"

#include "plm_register_loadable.h"

void
plm_register_loadable (void)
{
    FILE *fp;
    fp = fopen ("/tmp/plm_register_loadable.txt", "w");
    fprintf (fp, "Hello world\n");
    fclose (fp);
}
