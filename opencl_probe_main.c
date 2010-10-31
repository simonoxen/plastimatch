/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "delayload.h"
#include "opencl_probe.h"

int
main (int argc, char* argv[])
{
    int opencl_works;

    opencl_works = opencl_probe ();
    if (opencl_works) {
	printf ("Opencl works ok.\n");
    } else {
	printf ("Opencl does not work.\n");
    }
    return 0;
}
