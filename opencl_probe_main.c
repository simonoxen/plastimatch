/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>

#include "delayload.h"
#include "opencl_probe.h"

int
main (int argc, char* argv[])
{
    int opencl_works;

    LOAD_LIBRARY (libplmopencl);
    LOAD_SYMBOL_SPECIAL (opencl_probe, libplmopencl, int);

    if (!delayload_opencl()) { exit (0); };

    opencl_works = opencl_probe ();

    if (opencl_works) {
        printf ("Opencl works ok.\n");
    } else {
        printf ("Opencl does not work.\n");
    }

    UNLOAD_LIBRARY (libplmopencl);

    return 0;
}
