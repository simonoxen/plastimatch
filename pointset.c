/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "pointset.h"
#include "math_util.h"
#include "print_and_exit.h"

Pointset*
pointset_create (void)
{
    Pointset *ps;
    ps = (Pointset*) malloc (sizeof (Pointset));
    memset (ps, 0, sizeof (Pointset));
    return ps;
}

void
pointset_destroy (Pointset *ps)
{
    if (ps->points) {
	free (ps->points);
    }
    free (ps);
}

static Pointset *
pointset_load_fcsv (const char *fn)
{
    FILE *fp;
    Pointset *ps;
    char s[1024];

    fp = fopen (fn, "r");
    if (!fp) {
	return 0;
    }

    /* Check if this file is an fcsv file */
    fgets (s, 1024, fp);
    if (strncmp (s, "# Fiducial List file", strlen ("# Fiducial List file")))
    {
	fclose (fp);
	return 0;
    }

    /* Got an fcsv file.  Parse it. */
    ps = pointset_create ();
    while (!feof(fp)) {
	char *s2;
	float lm[3];
	int land_sel, land_vis;
	int rc;

        fgets (s, 1024, fp);
	if (feof(fp)) break;
        if (s[0]=='#') continue;

	// skip the label field assuming it does not contain commas
        s2 = strchr(s,',');
	
        rc = sscanf (s2, ",%f,%f,%f,%d,%d\n", 
	    &lm[0], &lm[1], &lm[2], &land_sel, &land_vis);
	if (rc != 5) {
	    print_and_exit ("Error parsing landmark file: %s\n", fn);
	}
	ps->num_points ++;
	pointset_resize (ps, ps->num_points);

	/* Note: Slicer landmarks are in RAS coordinates. 
	   Change RAS to LPS (note that LPS == ITK RAI). */
	ps->points[(ps->num_points-1)*3 + 0] = - lm[0];
	ps->points[(ps->num_points-1)*3 + 1] = - lm[1];
	ps->points[(ps->num_points-1)*3 + 2] = lm[2];
    }
    fclose (fp);

    return ps;
}

static Pointset *
pointset_load_txt (const char *fn)
{
    FILE *fp;
    Pointset *ps;
    char s[1024];

    fp = fopen (fn, "r");
    if (!fp) {
	return 0;
    }

    /* Parse as txt file */
    ps = pointset_create ();
    while (!feof(fp)) {
	float lm[3];
	int rc;

        fgets (s, 1024, fp);
	if (feof(fp)) break;
        if (s[0]=='#') continue;

        rc = sscanf (s, "%f , %f , %f\n", &lm[0], &lm[1], &lm[2]);
	if (rc != 3) {
	    rc = sscanf (s, "%f %f %f\n", &lm[0], &lm[1], &lm[2]);
	}
	if (rc != 3) {
	    print_and_exit ("Error parsing landmark file: %s\n", fn);
	}
	ps->num_points ++;
	pointset_resize (ps, ps->num_points);

	/* Assume LPS */
	ps->points[(ps->num_points-1)*3 + 0] = lm[0];
	ps->points[(ps->num_points-1)*3 + 1] = lm[1];
	ps->points[(ps->num_points-1)*3 + 2] = lm[2];
    }
    fclose (fp);

    return ps;
}

Pointset*
pointset_load (const char *fn)
{
    Pointset *ps;

    /* First try to load fcsv */
    ps = pointset_load_fcsv (fn);
    if (ps) return ps;

    /* If that doesn't work, try loading ASCII */
    ps = pointset_load_txt (fn);
    return ps;
}

void
pointset_resize (Pointset *ps, int new_size)
{
    ps->num_points = new_size;
    ps->points = (float*) realloc (ps->points, 
	    3 * (ps->num_points) * sizeof(float));
}
