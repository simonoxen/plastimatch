/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string.h>
#define HAVE_CONFIG_H 1               /* Needed for debian */
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dcuid.h"
#include "dcmtk/ofstd/ofstream.h"

#include "dicom_uid.h"

void
print_usage (void)
{
    fprintf (stderr, "Usage: dicom_uid [prefix]\n");
    exit (-1);
}

int
main (int argc, char* argv[])
{
    char uid[100];
    const char* uid_root = SITE_INSTANCE_UID_ROOT;

    if (argc == 2) {
	uid_root = argv[1];
    } else if (argc != 1) {
	print_usage ();
    }

    if (strlen (uid_root) >= 32) {
	fprintf (stderr, "Sorry, uid prefix should be less than 32 characters\n");
	exit (-1);
    }

    plm_generate_dicom_uid (uid, uid_root);

    printf ("%s\n", uid);
    return (0);
}
