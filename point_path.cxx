/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* =======================================================================*
   Output file format
        0 x1 y1 z1
        1 x2 y2 z2
        2 x3 y3 z3
        ...
        9 x9 y9 z9 
 * =======================================================================*/
#include "plm_config.h"
#include <stdio.h>
#include <string.h>
#include "tps.h"
#include "xform.h"

int input_phase;
int reference_phase = 5;
float input_position[3];
char deform_dir[256];
char output_file[256];

void
load_tps (struct tps_node** tps, int* num_tps_nodes, char* fn)
{
    FILE* fp;
    char buf[256];

    /* If file doesn't exist, then no tps for this phase */
    fp = fopen (fn, "r");
    if (!fp) return;

    while (fgets (buf, 256, fp)) {
	int rc;
	(*tps) = (struct tps_node*) realloc ((void*) (*tps), 
	    ((*num_tps_nodes)+1) * sizeof (struct tps_node));
	if (!(*tps)) {
	    fprintf (stderr, "Error allocating memory");
	    exit (-1);
	}
	rc = sscanf (buf, "%g %g %g %g %g %g %g", 
	    &(*tps)[(*num_tps_nodes)].src_pos[0],
	    &(*tps)[(*num_tps_nodes)].src_pos[1],
	    &(*tps)[(*num_tps_nodes)].src_pos[2],
	    &(*tps)[(*num_tps_nodes)].tgt_pos[0],
	    &(*tps)[(*num_tps_nodes)].tgt_pos[1],
	    &(*tps)[(*num_tps_nodes)].tgt_pos[2],
	    &(*tps)[(*num_tps_nodes)].alpha
	    );
	if (rc != 7) {
	    fprintf (stderr, "Ill-formed input file: %s\n", fn);
	    exit (-1);
	}
	(*num_tps_nodes)++;
    }
    fclose (fp);
}

void
transform_point_bsp (Xform* xf, float pos[3])
{
    DoublePointType itk_1;
    DoublePointType itk_2;

    itk_1[0] = pos[0];
    itk_1[1] = pos[1];
    itk_1[2] = pos[2];
    itk_2 = xf->get_bsp()->TransformPoint (itk_1);
    pos[0] = itk_2[0];
    pos[1] = itk_2[1];
    pos[2] = itk_2[2];
}

/* Modify pos from original to deformed position */
void
compute_point_path (float pos[3], int output_phase)
{
    char fn[256];
    Xform xf;
    struct tps_node* tps = 0;
    int num_tps_nodes = 0;

    if (input_phase == output_phase) return;

    /* TPS of input */
    sprintf (fn, "%s/%d_rbf_inv.txt", deform_dir, input_phase);
    load_tps (&tps, &num_tps_nodes, fn);
    tps_transform_point (tps, num_tps_nodes, pos);
    tps_free (&tps, &num_tps_nodes);

    /* BSP from input to reference */
    if (input_phase != reference_phase) {
	sprintf (fn, "%s/t_%d_%d_bsp.txt", deform_dir, input_phase, reference_phase);
	load_xform (&xf, fn);
	transform_point_bsp (&xf, pos);
    }

    if (output_phase != reference_phase) {
	sprintf (fn, "%s/t_%d_%d_bsp.txt", deform_dir, reference_phase, output_phase);
	load_xform (&xf, fn);
	transform_point_bsp (&xf, pos);
    }

    /* TPS of output */
    sprintf (fn, "%s/%d_rbf_fwd.txt", deform_dir, output_phase);
    load_tps (&tps, &num_tps_nodes, fn);
    tps_transform_point (tps, num_tps_nodes, pos);
    tps_free (&tps, &num_tps_nodes);
}

int
main (int argc, char *argv[])
{
    int i;
    FILE* fp;

    if (argc != 5) {
	fprintf (stderr, "Usage: point_path deform_dir phase \"point\" output_file\n");
	return 1;
    }
    if (strlen(argv[1]) > 256) {
	fprintf (stderr, "Sorry, deform_dir path must be shorter than 256 characters\n");
	return 1;
    }
    strcpy (deform_dir, argv[1]);
    if (1 != sscanf (argv[2], "%d", &input_phase)) {
	fprintf (stderr, "Error parsing phase argument\n");
	return 1;
    }
    if (3 != sscanf (argv[3], "%g %g %g", &input_position[0], &input_position[1], &input_position[2])) {
	fprintf (stderr, "Error parsing point argument\n");
	return 1;
    }
    if (strlen(argv[4]) > 256) {
	fprintf (stderr, "Sorry, output_file path must be shorter than 256 characters\n");
	return 1;
    }
    strcpy (output_file, argv[4]);

    fp = fopen (output_file, "wb");
    if (!fp) {
	fprintf (stderr, "Couldn't open file \"%s\" for read\n", output_file);
	return 1;
    }
    for (i = 0; i < 10; i++) {
	float deformed_position[3];
	memcpy (deformed_position, input_position, 3 * sizeof(float));
	compute_point_path (deformed_position, i);
	fprintf (fp, "%d %g %g %g\n", i, deformed_position[0], 
		deformed_position[1], deformed_position[2]);
    }
    fclose (fp);
    return 0;
}
