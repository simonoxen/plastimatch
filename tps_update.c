/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    Copyright 2007
    Massachusetts General Hospital
    All rights reserved

   Two input files, of format:
        0 x1 y1 z1
        1 x2 y2 z2
        2 x3 y3 z3
        ...
        9 x9 y9 z9

   Phase files which have changed are updated like this
        x y z dx dy dz a
        
    ----------------------------------------------------------------------- */
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BUFLEN 1024
#define MOTION_TOL 0.01
#define DIST_MULTIPLIER 5.0

char tgt_fn[256];
char src_fn[256];
char deform_dir[256];
char tps_fn[256];

int
main (int argc, char *argv[])
{
    int i;
    FILE *src_fp, *tgt_fp, *tps_fp;

    /* Parse arguments */
    if (argc != 4) {
	fprintf (stderr, "Usage: tps_update src_fn tgt_fn deform_dir\n");
	return 1;
    }
    if (strlen(argv[1]) > 256) {
	fprintf (stderr, "Sorry, src_fn path must be shorter than 256 characters\n");
	return 1;
    }
    strcpy (src_fn, argv[1]);
    if (strlen(argv[2]) > 256) {
	fprintf (stderr, "Sorry, tgt_fn path must be shorter than 256 characters\n");
	return 1;
    }
    strcpy (tgt_fn, argv[2]);
    if (strlen(argv[3]) > 256) {
	fprintf (stderr, "Sorry, deform_dir path must be shorter than 256 characters\n");
	return 1;
    }
    strcpy (deform_dir, argv[3]);

    src_fp = fopen (src_fn, "r");
    if (!src_fp) {
	fprintf (stderr, "Couldn't open file \"%s\" for read\n", src_fn);
	return 1;
    }
    tgt_fp = fopen (tgt_fn, "r");
    if (!tgt_fp) {
	fprintf (stderr, "Couldn't open file \"%s\" for read\n", tgt_fn);
	return 1;
    }

    /* Loop through points in source file */
    for (i = 0; i < 10; i++) {
	int rc;
	char src_buf[BUFLEN], tgt_buf[BUFLEN];
	int src_phase, tgt_phase;
	float src_pos[3], tgt_pos[3];
	if (!fgets (src_buf, BUFLEN, src_fp)) {
	    fprintf (stderr, "Ill-formed input file (1): %s\n", src_fn);
	    return 1;
	}
	if (!fgets (tgt_buf, BUFLEN, tgt_fp)) {
	    fprintf (stderr, "Ill-formed input file (1): %s\n", tgt_fn);
	    return 1;
	}
	rc = sscanf (src_buf, "%d %g %g %g", &src_phase, &src_pos[0], 
		    &src_pos[1],  &src_pos[2]);
	if (rc != 4 || src_phase != i) {
	    fprintf (stderr, "Ill-formed input file (2): %s\n", src_fn);
	    return 1;
	}
	rc = sscanf (tgt_buf, "%d %g %g %g", &tgt_phase, &tgt_pos[0], 
		    &tgt_pos[1],  &tgt_pos[2]);
	if (rc != 4 || tgt_phase != i) {
	    fprintf (stderr, "Ill-formed input file (2): %s\n", tgt_fn);
	    return 1;
	}

	if ((fabs (src_pos[0] - tgt_pos[0]) > MOTION_TOL) ||
	    (fabs (src_pos[1] - tgt_pos[1]) > MOTION_TOL) ||
	    (fabs (src_pos[2] - tgt_pos[2]) > MOTION_TOL)) {
	    float dist;
	    char tps_fn[256];

	    dist = sqrt (((src_pos[0] - tgt_pos[0]) * (src_pos[0] - tgt_pos[0]))
		    + ((src_pos[1] - tgt_pos[1]) * (src_pos[1] - tgt_pos[1]))
		    + ((src_pos[2] - tgt_pos[2]) * (src_pos[2] - tgt_pos[2])));

	    /* Append to fwd deformation */
	    snprintf (tps_fn, 256, "%s/%d_rbf_fwd.txt", deform_dir, i);
	    tps_fp = fopen (tps_fn, "ab");
	    if (!tps_fp) {
		fprintf (stderr, "Couldn't open file \"%s\" for read\n", tps_fn);
		return 1;
	    }
	    fprintf (tps_fp, "%g %g %g %g %g %g %g\n", 
		    src_pos[0],
		    src_pos[1],
		    src_pos[2],
		    tgt_pos[0]-src_pos[0],
		    tgt_pos[1]-src_pos[1],
		    tgt_pos[2]-src_pos[2],
		    DIST_MULTIPLIER * dist
		    );
	    fclose (tps_fp);

	    /* Append to inv deformation */
	    snprintf (tps_fn, 256, "%s/%d_rbf_inv.txt", deform_dir, i);
	    tps_fp = fopen (tps_fn, "ab");
	    if (!tps_fp) {
		fprintf (stderr, "Couldn't open file \"%s\" for read\n", tps_fn);
		return 1;
	    }
	    fprintf (tps_fp, "%g %g %g %g %g %g %g\n", 
		    tgt_pos[0],
		    tgt_pos[1],
		    tgt_pos[2],
		    src_pos[0]-tgt_pos[0],
		    src_pos[1]-tgt_pos[1],
		    src_pos[2]-tgt_pos[2],
		    DIST_MULTIPLIER * dist
		    );
	    fclose (tps_fp);
	}
    }

    fclose (src_fp);
    fclose (tgt_fp);
    return 0;
}
