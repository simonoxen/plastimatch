/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <io.h>        // windows //
#endif
#include "plm_int.h"
#include "volume.h"

#define LINELEN 128
#define MIN_SHORT -32768
#define MAX_SHORT 32767

#define WRITE_BLOCK (1024*1024)

/* GCS Jun 18, 2008.  When using MSVC 2005, large fwrite calls to files over 
    samba mount fails.  This seems to be a bug in the C runtime library.
    This function works around the problem by breaking up the large write 
    into many "medium-sized" writes. */
void 
fwrite_block (void* buf, size_t size, size_t count, FILE* fp)
{
    size_t left_to_write = count * size;
    size_t cur = 0;
    char* bufc = (char*) buf;

    while (left_to_write > 0) {
	size_t this_write, rc;

	this_write = left_to_write;
	if (this_write > WRITE_BLOCK) this_write = WRITE_BLOCK;
	rc = fwrite (&bufc[cur], 1, this_write, fp);
	if (rc != this_write) {
	    fprintf (stderr, 
		"Error writing to file. rc=%u, this_write=%u\n",
		(uint32_t) rc, (uint32_t) this_write);
	    return;
	}
	cur += rc;
	left_to_write -= rc;
    }
}
