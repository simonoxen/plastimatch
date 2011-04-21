/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "string_util.h"

void
string_util_rtrim_whitespace (char *s)
{
    int len = strlen (s);
    while (len > 0 && isspace(s[len-1])) {
	s[len-1] = 0;
	len--;
    }
}

int
parse_int13 (int *arr, const char *string)
{
    int rc;
    rc = sscanf (string, "%d %d %d", &arr[0], &arr[1], &arr[2]);
    if (rc == 3) {
	return 0;
    } else if (rc == 1) {
	arr[1] = arr[2] = arr[0];
	return 0;
    } else {
	/* Failure */
	return 1;
    }
}
