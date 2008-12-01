/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"
#include "itkImage.h"
#include "itkInverseDeformationFieldImageFilter.h"
#include "itkIterativeInverseDeformationFieldImageFilter.h"
#include "getopt.h"
#include "vf_invert.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "xform.h"

void
vf_invert_main (Vf_Invert_Parms* parms)
{
    typedef itk::InverseDeformationFieldImageFilter < DeformationFieldType, DeformationFieldType >  FilterType;
    
    PlmImageHeader pih;

    if (parms->fixed_img_fn[0]) {
	/* if given, use the parameters from user-supplied fixed image */
	FloatImageType::Pointer fixed = load_float (parms->fixed_img_fn);
	pih.set_from_itk_image (fixed);
    } else {
	pih.set_from_gpuit (parms->origin, parms->spacing, parms->dim);
    }

    FilterType::Pointer filter = FilterType::New ();
    DeformationFieldType::Pointer vf_in = load_float_field (parms->vf_in_fn);
    filter->SetInput (vf_in);
    filter->SetOutputOrigin (pih.m_origin);
    filter->SetOutputSpacing (pih.m_spacing);
    filter->SetSize (pih.m_region.GetSize());

    //filter->SetOutsideValue( 0 );
    filter->Update();
    DeformationFieldType::Pointer vf_out = filter->GetOutput();
    save_image (vf_out, parms->vf_out_fn);
}


void
print_usage (void)
{
    printf ("Usage: vf_invert --input=vf_in --output=vf_out\n");
    printf ("           --dims=\"x y z\" --origin=\"x y z\" --spacing=\"x y z\"\n");
    printf ("       ||  --fixed=\"fixed-img\"\n");
    exit (-1);
}

void
parse_args (Vf_Invert_Parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           1 },
	{ "output",         required_argument,      NULL,           2 },
	{ "dims",           required_argument,      NULL,           3 },
	{ "origin",         required_argument,      NULL,           4 },
	{ "spacing",        required_argument,      NULL,           5 },
	{ "fixed",          required_argument,      NULL,           6 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->vf_in_fn, optarg, _MAX_PATH);
	    break;
	case 2:
	    strncpy (parms->vf_out_fn, optarg, _MAX_PATH);
	    break;
	case 3: {
		rc = sscanf (optarg, "%d %d %d", &(parms->dim[0]), 
			&(parms->dim[1]), &(parms->dim[2]));
		if (rc != 3) {
		    print_usage();
		}
	    }
	    break;
	case 4:
	    rc = sscanf (optarg, "%g %g %g", &(parms->origin[0]), 
		    &(parms->origin[1]), &(parms->origin[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 5:
	    rc = sscanf (optarg, "%g %g %g", &(parms->spacing[0]), 
		    &(parms->spacing[1]), &(parms->spacing[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 6:
	    strncpy (parms->fixed_img_fn, optarg, _MAX_PATH);
	    break;
	default:
	    break;
	}
    }
    if (!parms->vf_in_fn[0] || !parms->vf_out_fn[0] || ((!parms->dim[0] || !parms->origin[0] || !parms->spacing[0]) && (!parms->fixed_img_fn[0]))) {
	printf ("Error: must specify all options\n");
	print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Vf_Invert_Parms parms;
    
    parse_args (&parms, argc, argv);

    vf_invert_main (&parms);

    printf ("Finished!\n");
    return 0;
}
