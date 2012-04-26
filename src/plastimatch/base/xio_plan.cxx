/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "plmsys.h"

#include "bstrlib.h"
#include "bstrwrap.h"
#include "xio_plan.h"

void
xio_plan_get_studyset (const char *filename, char *studyset)
{
    FILE *fp;

    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }
    CBStream bs ((bNread) fread, fp);

    /* Get version string
       0062101a - xio version 4.33.02
       006d101a - xio version 4.50 */
    CBString version = bs.readLine ('\n');
    printf ("Version = %s\n", (const char*) version);
    int rc, version_int;
    rc = sscanf ((const char*) version, "%x", &version_int);
    if (rc != 1) {
	/* Couldn't parse version string -- default to older format. */
	version_int = 0x62101a;
    }
    printf ("rc = %d, version_int = 0x%x\n", rc, version_int);

    /* Skip 4 lines for xio 4.33.02, skip 5 lines for xio 4.50. */
    bs.readLine ('\n');
    bs.readLine ('\n');
    bs.readLine ('\n');
    bs.readLine ('\n');
    if (version_int > 0x62101a) {
	bs.readLine ('\n');
    }

    /* Read studyset name */
    CBString line1 = bs.readLine ('\n');
    strcpy (studyset, (const char *) line1);
    studyset[strlen (studyset) - 1] = '\0';

    fclose (fp);
}
