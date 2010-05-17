/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bstrlib.h"

#include "print_and_exit.h"
#include "xio_plan.h"

void
xio_plan_get_studyset (const char *filename, char *studyset)
{
    FILE *fp;

    struct bStream * bs;
    bstring line1 = bfromcstr ("");

    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    bs = bsopen ((bNread) fread, fp);

    /* Skip 5 lines */
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');

    /* Read studyset name */
    bsreadln (line1, bs, '\n');
    strcpy (studyset, (char *) line1->data);
    studyset[strlen (studyset) - 1] = '\0';
}
