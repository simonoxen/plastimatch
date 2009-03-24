/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Correct mha files which have incorrect patient orientations */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "plm_config.h"
#include "mathutil.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "synthetic_mha_main.h"
#include "itk_image.h"
#include "getopt.h"

void
do_synthetic_mha (Synthetic_mha_parms* parms)
{
    /* Create ITK image */
    FloatImageType::SizeType sz;
    FloatImageType::IndexType st;
    FloatImageType::RegionType rg;
    FloatImageType::PointType og;
    FloatImageType::SpacingType sp;
    FloatImageType::DirectionType dc;
    for (int d1 = 0; d1 < 3; d1++) {
	st[d1] = 0;
	sz[d1] = parms->res[d1];
	sp[d1] = parms->volume_size[d1] / ((float) parms->res[d1]);
	og[d1] = parms->origin[d1];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);

    FloatImageType::Pointer im_out = FloatImageType::New();
    im_out->SetRegions(rg);
    im_out->SetOrigin(og);
    im_out->SetSpacing(sp);
    im_out->Allocate();

    /* Iterate through image, setting values */
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > IteratorType;
    IteratorType it_out (im_out, im_out->GetRequestedRegion());
    for (it_out.GoToBegin(); !it_out.IsAtEnd(); ++it_out) {
	FloatPointType phys;
	float f = 0.0f;

	FloatImageType::IndexType idx = it_out.GetIndex ();
	im_out->TransformIndexToPhysicalPoint (idx, phys);
	switch (parms->pattern) {
	case PATTERN_GAUSS:
	    f = 0;
	    for (int d = 0; d < 3; d++) {
		float f1 = phys[d] - parms->gauss_center[d];
		f1 = f1 / parms->gauss_std[d];
		f += f1 * f1;
	    }
	    f = exp (-0.5 * f);	    /* f \in (0,1] */
	    f = (1 - f) * parms->background + f * parms->foreground;
	    break;
	case PATTERN_RECT:
	    if (phys[0] >= parms->rect_size[0] 
		&& phys[0] <= parms->rect_size[1] 
		&& phys[1] >= parms->rect_size[2] 
		&& phys[1] <= parms->rect_size[3] 
		&& phys[2] >= parms->rect_size[4] 
		&& phys[2] <= parms->rect_size[5])
	    {
		f = parms->foreground;
	    } else {
		f = parms->background;
	    }
	    break;
	case PATTERN_SPHERE:
	    f = 0;
	    for (int d = 0; d < 3; d++) {
		float f1 = phys[d] - parms->sphere_center[d];
		f1 = f1 / parms->sphere_radius[d];
		f += f1 * f1;
	    }
	    if (f > 1.0) {
		f = parms->background;
	    } else {
		f = parms->foreground;
	    }
	    break;
	default:
	    f = 0.0f;
	    break;
	}
	it_out.Set (f);
    }

    /* Save the file */
    save_float (im_out, parms->output_fn);
}

void
print_usage (void)
{
    printf ("Usage: resample_mha [options]\n");
    printf ("Required:   --output=file\n"
	    "Optional:   --output-type={uchar,short,ushort,float}\n"
	    "            --pattern={gauss,rect}\n"
	    "            --origin=\"x y z\"\n"
	    "            --resolution=\"x [y z]\"\n"
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
parse_args (Synthetic_mha_parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "output",         required_argument,      NULL,           1 },
	{ "output_type",    required_argument,      NULL,           2 },
	{ "output-type",    required_argument,      NULL,           2 },
	{ "pattern",        required_argument,      NULL,           3 },
	{ "origin",         required_argument,      NULL,           4 },
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
	    if (!strcmp(optarg,"ushort") || !strcmp(optarg,"unsigned")) {
		parms->output_type = PLM_IMG_TYPE_ITK_USHORT;
	    }
	    else if (!strcmp(optarg,"short") || !strcmp(optarg,"signed")) {
		parms->output_type = PLM_IMG_TYPE_ITK_SHORT;
	    }
	    else if (!strcmp(optarg,"float")) {
		parms->output_type = PLM_IMG_TYPE_ITK_FLOAT;
	    }
	    else if (!strcmp(optarg,"mask") || !strcmp(optarg,"uchar")) {
		parms->output_type = PLM_IMG_TYPE_ITK_UCHAR;
	    }
#if defined (commentout)
	    else if (!strcmp(optarg,"vf")) {
		parms->output_type = PLM_IMG_TYPE_ITK_FLOAT_FIELD;
	    }
#endif
	    else {
		print_usage();
	    }
	    break;
	case 3:
	    if (!strcmp (optarg, "gauss")) {
		parms->pattern = PATTERN_GAUSS;
	    }
	    else if (!strcmp (optarg, "rect")) {
		parms->pattern = PATTERN_RECT;
	    }
	    else if (!strcmp (optarg, "sphere")) {
		parms->pattern = PATTERN_SPHERE;
	    }
	    else {
		print_usage();
	    }
	    break;
	case 4:
	    rc = sscanf (optarg, "%g %g %g", 
			 &(parms->origin[0]), 
			 &(parms->origin[1]), 
			 &(parms->origin[2]));
	    if (rc != 3) {
		printf ("Origin option must have three arguments\n");
		exit (1);
	    }
	    parms->have_origin = 1;
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
			 &(parms->res[0]), 
			 &(parms->res[1]), 
			 &(parms->res[2]));
	    if (rc == 1) {
		parms->res[1] = parms->res[0];
		parms->res[2] = parms->res[0];
	    } else if (rc != 3) {
		printf ("Resolution option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 7:
	    rc = sscanf (optarg, "%g", &(parms->background));
	    if (rc != 1) {
		printf ("Background option must have one arguments\n");
		exit (1);
	    }
	    break;
	case 8:
	    rc = sscanf (optarg, "%g", &(parms->foreground));
	    if (rc != 1) {
		printf ("Foreground option must have one arguments\n");
		exit (1);
	    }
	    break;
	case 9:
	    rc = sscanf (optarg, "%g %g %g", 
			 &(parms->gauss_center[0]), 
			 &(parms->gauss_center[1]), 
			 &(parms->gauss_center[2]));
	    if (rc != 3) {
		printf ("Gauss_center option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 10:
	    rc = sscanf (optarg, "%g %g %g", 
			 &(parms->gauss_std[0]), 
			 &(parms->gauss_std[1]), 
			 &(parms->gauss_std[2]));
	    if (rc == 1) {
		parms->gauss_std[1] = parms->gauss_std[0];
		parms->gauss_std[2] = parms->gauss_std[0];
	    }
	    else if (rc != 3) {
		printf ("Gauss_std option must have one or three arguments\n");
		exit (1);
	    }
	    break;
	case 11:
	    rc = sscanf (optarg, "%g %g %g %g %g %g", 
			 &(parms->rect_size[0]), 
			 &(parms->rect_size[1]), 
			 &(parms->rect_size[2]), 
			 &(parms->rect_size[3]), 
			 &(parms->rect_size[4]), 
			 &(parms->rect_size[5]));
	    if (rc == 1) {
		parms->rect_size[0] = - 0.5 * parms->rect_size[0];
		parms->rect_size[1] = - parms->rect_size[0];
		parms->rect_size[2] = + parms->rect_size[0];
		parms->rect_size[3] = - parms->rect_size[0];
		parms->rect_size[4] = + parms->rect_size[0];
		parms->rect_size[5] = - parms->rect_size[0];
	    }
	    else if (rc == 3) {
		parms->rect_size[0] = - 0.5 * parms->rect_size[0];
		parms->rect_size[2] = - 0.5 * parms->rect_size[1];
		parms->rect_size[4] = - 0.5 * parms->rect_size[2];
		parms->rect_size[1] = - parms->rect_size[0];
		parms->rect_size[3] = - parms->rect_size[2];
		parms->rect_size[5] = - parms->rect_size[4];
	    }
	    else if (rc != 6) {
		printf ("Rect_size option must have one, three, or six arguments\n");
		exit (1);
	    }
	    break;
	case 12:
	    rc = sscanf (optarg, "%g %g %g", 
			 &(parms->sphere_center[0]), 
			 &(parms->sphere_center[1]), 
			 &(parms->sphere_center[2]));
	    if (rc != 3) {
		printf ("Sphere center option must have three arguments\n");
		exit (1);
	    }
	    break;
	case 13:
	    rc = sscanf (optarg, "%g %g %g", 
			 &(parms->sphere_radius[0]), 
			 &(parms->sphere_radius[1]), 
			 &(parms->sphere_radius[2]));
	    if (rc == 1) {
		parms->sphere_radius[1] = parms->sphere_radius[0];
		parms->sphere_radius[2] = parms->sphere_radius[0];
	    }
	    else if (rc != 3) {
		printf ("Sphere_radius option must have one or three arguments\n");
		exit (1);
	    }
	    break;
	default:
	    break;
	}
    }
    if (!parms->output_fn[0]) {
	printf ("Error: must specify --output\n");
	print_usage();
    }

    /* If origin not specified, volume is centered about size */
    if (!parms->have_origin) {
	int d;
	for (d = 0; d < 3; d++) {
	    parms->origin[d] = - 0.5 * parms->volume_size[d] 
		    + 0.5 * parms->volume_size[d] / parms->res[d];
	}
    }
}

int 
main (int argc, char* argv[])
{
    Synthetic_mha_parms parms;

    parse_args (&parms, argc, argv);

    do_synthetic_mha (&parms);

    return 0;
}
