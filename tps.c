/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tps.h"
#include "print_and_exit.h"

#define BUFLEN 1024
#define MOTION_TOL 0.01
#define DIST_MULTIPLIER 5.0

char tgt_fn[256];
char src_fn[256];
char deform_dir[256];
char tps_fn[256];

Tps_xform*
tps_xform_alloc (void)
{
    Tps_xform *tps_xform;

    tps_xform = (Tps_xform*) malloc (sizeof (Tps_xform));
    if (!tps_xform) {
	print_and_exit ("Out of memory\n");
    }
    memset (tps_xform, 0, sizeof (Tps_xform));
    return tps_xform;
}

Tps_xform*
tps_load (char* fn)
{
    char buf[256];
    FILE *fp;
    Tps_xform *tps_xform;

    tps_xform = tps_xform_alloc ();

    /* If file doesn't exist, then no tps for this phase */
    fp = fopen (fn, "r");
    if (!fp) return tps_xform;

    while (fgets (buf, 256, fp)) {
	int rc;
	Tps_node *curr_node;
	tps_xform->tps_nodes = (struct tps_node*) 
		realloc ((void*) (tps_xform->tps_nodes), 
			 (tps_xform->num_tps_nodes + 1) * sizeof (Tps_node));
	if (!(tps_xform->tps_nodes)) {
	    print_and_exit ("Error allocating memory");
	}
	curr_node = &tps_xform->tps_nodes[tps_xform->num_tps_nodes];
	rc = sscanf (buf, "%g %g %g %g %g %g %g", 
		     &curr_node->src_pos[0],
		     &curr_node->src_pos[1],
		     &curr_node->src_pos[2],
		     &curr_node->tgt_pos[0],
		     &curr_node->tgt_pos[1],
		     &curr_node->tgt_pos[2],
		     &curr_node->alpha
		     );
	if (rc != 7) {
	    print_and_exit ("Ill-formed input file: %s\n", fn);
	}
	tps_xform->num_tps_nodes++;
    }
    fclose (fp);
    return tps_xform;
}

void
tps_free (Tps_node** tps, int *num_tps_nodes)
{
    free (*tps);
    *tps = 0;
    *num_tps_nodes = 0;
}

void
tps_transform_point (Tps_node* tps, int num_tps_nodes, float pos[3])
{
    int i, j;
    float new_pos[3];

    if (!tps) return;

    memcpy (new_pos, pos, 3 * sizeof(float));
    for (i = 0; i < num_tps_nodes; i++) {
	float dist = sqrt (
	    ((pos[0] - tps[i].src_pos[0]) * (pos[0] - tps[i].src_pos[0])) +
	    ((pos[1] - tps[i].src_pos[1]) * (pos[1] - tps[i].src_pos[1])) +
	    ((pos[2] - tps[i].src_pos[2]) * (pos[2] - tps[i].src_pos[2])));
	dist = dist / tps[i].alpha;
	if (dist < 1.0) {
	    float weight = (1 - dist) * (1 - dist);
	    for (j = 0; j < 3; j++) {
		new_pos[j] += weight * tps[i].tgt_pos[j];
	    }
	}
    }

    memcpy (pos, new_pos, 3 * sizeof(float));
}

#if defined (commentout)
int
main (int argc, char *argv[])
{
    int i;
    FILE *src_fp, *tgt_fp, *tps_fp;

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
#endif
