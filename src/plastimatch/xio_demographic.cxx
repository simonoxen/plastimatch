/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bstrlib.h"
#include "bstring_util.h"
#include "print_and_exit.h"
#include "xio_demographic.h"

Xio_demographic::Xio_demographic (const char *filename)
{
    FILE *fp;
    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }
    CBStream bs ((bNread) fread, fp);

    /* version string 
       00011017 - pxio version 4.2 */
    CBString version = bs.readLine ('\n');

    /* date (for what?) */
    CBString date = bs.readLine ('\n');

    /* important stuff here */
    m_patient_name = bs.readLine ('\n');
    m_patient_name.rtrim();
    m_patient_id = bs.readLine ('\n');
    m_patient_id.rtrim();

    fclose (fp);
}

Xio_demographic::~Xio_demographic ()
{
}
