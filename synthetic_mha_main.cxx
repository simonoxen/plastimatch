/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Correct mha files which have incorrect patient orientations */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "plm_config.h"
#include "math_util.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "synthetic_mha_main.h"
#include "itk_image.h"
#include "getopt.h"

void
do_synthetic_mha (char* fn, Synthetic_mha_parms* parms)
{
    /* Create image */
    FloatImageType::Pointer img = synthetic_mha (parms);

    /* Save to file */
    switch (parms->output_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	itk_image_save_uchar (img, fn);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	itk_image_save_short (img, fn);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	itk_image_save_ushort (img, fn);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	itk_image_save_uint32 (img, fn);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save_float (img, fn);
	break;
    }
}

void
print_usage (void)
{
    printf ("Usage: resample_mha [options]\n");
    printf ("Required:   --output=file\n"
	    "Optional:   --output-type={uchar,short,ushort,ulong,float}\n"
	    "            --pattern={gauss,rect,sphere}\n"
	    "            --origin=\"x y z\"\n"
	    "            --resolution=\"x [y z]\"\n"
	    "            --spacing=\"x [y z]\"\n"
	    "            --volume-size=\"x [y z]\"\n"
	    "            --background=val\n"
	    "            --foreground=val\n"
	    "Gaussian:   --gauss-center=\"x y z\"\n"
	    "            --gauss-std=\"x [y z]\"\n"
	    "Rect:       --rect-size=\"x [y z]\"\n"
	    "            --rect-size=\"x1 x2 y1 y2 z1 z2\"\n"
	    "Sphere:     --sphere-center=\"x y z\"\n"
	    "            --sphere-radius=\"x [y z]\"\n"
	    );
    exit (1);
}

void
parse_args (Synthetic_mha_main_parms* parms, int argc, char* argv[])
{
    int ch, rc;
    Synthetic_mha_parms *sm_parms = &parms->sm_parms;

    static struct option longopts[] = {
	{ "output",         required_argument,      NULL,           1 },
	{ "output_type",    required_argument,      NULL,           2 },
	{ "output-type",    required_argument,      NULL,           2 },
	{ "pattern",        required_argument,      NULL,           3 },
	{ "origin",         required_argument,      NULL,           4 },
	{ "offset",         required_argument,      NULL,           4 },
	{ "volume_size",    required_argument,      NULL,           5 },
	{ "volume-size",    required_argument,      NULL,           5 },
	{ "res",            required_argument,      NULL,           6 },
	{ "resolution",     required_argument,      NULL,           6 },
	{ "background",     required_argument,      NULL,           7 },
	{ "foreground",     required_argument,      NULL,           8 },
	{ "gauss_center",   required_argument,      NULL,           9 },
	{ "gauss-center",   required_argument,      NULL,           9 },
	{ "gauss_std",      required_argument,      NULL,           10 },
	{ "gauss-std",      required_argument,      NULL,           10 },
	{ "rect_size",      required_argument,      NULL,           11 },
	{ "rect-size",      required_argument,      NULL,           11 },
	{ "sphere_center",  required_argument,      NULL,           12 },
	{ "sphere-center",  required_argument,      NULL,           12 },
	{ "sphere_radius",  required_argument,      NULL,           13 },
	{ "sphere-radius",  required_argument,      NULL,           13 },
	{ "spacing",        required_argument,      NULL,           14 },
	{ NULL,             0,                      NULL,           0 }
    };

    char **parm_argv = argv;
    int parm_argc = argc;

    while ((ch = getopt_long (parm_argc, parm_argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->output_fn, optarg, _MAX_PATH);
	    break;
	case 2:
	    sm_parms->output_type = plm_image_type_parse (optarg);
	    if (sm_parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
		printf ("Unknown output type option\n");
		print_usage();
	    }
	    break;
	case 3:
	    if (!strcmp (optarg, "gauss")) {
		sm_parms->pattern = PATTERN_GAUSS;
	    }
	    else if (!strcmp (optarg, "rect")) {
		sm_parms->pattern = PATTERN_RECT;
	    }
	    else if (!strcmp (optarg, "sphere")) {
		sm_parms->pattern = PATTERN_SPHERE;
	    }
	    else {
		printf ("Unknown pattern option\n");
		print_usage();
	    }
	    break;
	case 4:
	    rc = sscanf (optarg, "%g %g %g", 
		&(sm_parms->offset[0]), 
		&(sm_parms->offset[1]), 
		&(sm_parms->offset[2]));
	    if (rc != 3) {
		printf ("Origin option must have three arguments\n");
		exit (1);
	    }
	    parms->have_offset = 1;
	    break;
	case 5:
	    rc = sscanf (optarg, "%g %g %g", 
		&(parms->volume_size[0]), 
		&(parms->volume_size[1]), 
		&(parms->volume_size[2]));
	    if (rc == 1) {
		parms->volume_size[1] = parms->volume_size[0];
		parms->volume_size[2] = parms->volume_size[0];
	    } else if (rc != 3) {
		printf ("Volume_size option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 6:
	    rc = sscanf (optarg, "%d %d %d", 
		&(sm_parms->dim[0]), 
		&(sm_parms->dim[1]), 
		&(sm_parms->dim[2]));
	    if (rc == 1) {
		sm_parms->dim[1] = sm_parms->dim[0];
		sm_parms->dim[2] = sm_parms->dim[0];
	    } else if (rc != 3) {
		printf ("Resolution option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 7:
	    rc = sscanf (optarg, "%g", &(sm_parms->background));
	    if (rc != 1) {
		printf ("Background option must have one arguments\n");
		exit (1);
	    }
	    break;
	case 8:
	    rc = sscanf (optarg, "%g", &(sm_parms->foreground));
	    if (rc != 1) {
		printf ("Foreground option must have one arguments\n");
		exit (1);
	    }
	    break;
	case 9:
	    rc = sscanf (optarg, "%g %g %g", 
		&(sm_parms->gauss_center[0]), 
		&(sm_parms->gauss_center[1]), 
		&(sm_parms->gauss_center[2]));
	    if (rc != 3) {
		printf ("Gauss_center option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 10:
	    rc = sscanf (optarg, "%g %g %g", 
		&(sm_parms->gauss_std[0]), 
		&(sm_parms->gauss_std[1]), 
		&(sm_parms->gauss_std[2]));
	    if (rc == 1) {
		sm_parms->gauss_std[1] = sm_parms->gauss_std[0];
		sm_parms->gauss_std[2] = sm_parms->gauss_std[0];
	    }
	    else if (rc != 3) {
		printf ("Gauss_std option must have one or three arguments\n");
		exit (1);
	    }
	    break;
	case 11:
	    rc = sscanf (optarg, "%g %g %g %g %g %g", 
		&(sm_parms->rect_size[0]), 
		&(sm_parms->rect_size[1]), 
		&(sm_parms->rect_size[2]), 
		&(sm_parms->rect_size[3]), 
		&(sm_parms->rect_size[4]), 
		&(sm_parms->rect_size[5]));
	    if (rc == 1) {
		sm_parms->rect_size[0] = - 0.5 * sm_parms->rect_size[0];
		sm_parms->rect_size[1] = - sm_parms->rect_size[0];
		sm_parms->rect_size[2] = + sm_parms->rect_size[0];
		sm_parms->rect_size[3] = - sm_parms->rect_size[0];
		sm_parms->rect_size[4] = + sm_parms->rect_size[0];
		sm_parms->rect_size[5] = - sm_parms->rect_size[0];
	    }
	    else if (rc == 3) {
		sm_parms->rect_size[4] = - 0.5 * sm_parms->rect_size[2];
		sm_parms->rect_size[2] = - 0.5 * sm_parms->rect_size[1];
		sm_parms->rect_size[0] = - 0.5 * sm_parms->rect_size[0];
		sm_parms->rect_size[1] = - sm_parms->rect_size[0];
		sm_parms->rect_size[3] = - sm_parms->rect_size[2];
		sm_parms->rect_size[5] = - sm_parms->rect_size[4];
	    }
	    else if (rc != 6) {
		printf ("Rect_size option must have one, three, or six arguments\n");
		exit (1);
	    }
	    break;
	case 12:
	    rc = sscanf (optarg, "%g %g %g", 
		&(sm_parms->sphere_center[0]), 
		&(sm_parms->sphere_center[1]), 
		&(sm_parms->sphere_center[2]));
	    if (rc != 3) {
		printf ("Sphere center option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 13:
	    rc = sscanf (optarg, "%g %g %g", 
		&(sm_parms->sphere_radius[0]), 
		&(sm_parms->sphere_radius[1]), 
		&(sm_parms->sphere_radius[2]));
	    if (rc == 1) {
		sm_parms->sphere_radius[1] = sm_parms->sphere_radius[0];
		sm_parms->sphere_radius[2] = sm_parms->sphere_radius[0];
	    }
	    else if (rc != 3) {
		printf ("Sphere_radius option must have one or three arguments\n");
		exit (1);
	    }
	    break;
	case 14:
	    rc = sscanf (optarg, "%g %g %g", 
		&(sm_parms->spacing[0]), 
		&(sm_parms->spacing[1]), 
		&(sm_parms->spacing[2]));
	    if (rc == 1) {
		sm_parms->spacing[1] = sm_parms->spacing[0];
		sm_parms->spacing[2] = sm_parms->spacing[0];
	    }
	    else if (rc != 3) {
		printf ("Spacing option must have one or three arguments\n");
		exit (1);
	    }
	    parms->have_spacing = 1;
	default:
	    break;
	}
    }
    if (!parms->output_fn[0]) {
	printf ("Error: must specify --output\n");
	print_usage();
    }

    /* If origin not specified, volume is centered about size */
    if (!parms->have_offset) {
	int d;
	for (d = 0; d < 3; d++) {
	    sm_parms->offset[d] = - 0.5 * parms->volume_size[d] 
		+ 0.5 * parms->volume_size[d] / sm_parms->dim[d];
	}
    }

    /* Set spacing based on size and resolution */
    if (!parms->have_spacing) {
	for (int d1 = 0; d1 < 3; d1++) {
	    sm_parms->spacing[d1] 
		= parms->volume_size[d1] / ((float) sm_parms->dim[d1]);
	}
    }
}

int 
main (int argc, char* argv[])
{
    Synthetic_mha_main_parms parms;

    parse_args (&parms, argc, argv);

    do_synthetic_mha (parms.output_fn, &parms.sm_parms);

    return 0;
}
