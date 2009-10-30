/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"

#include "resample_mha_main.h"
#include "itk_image.h"
#include "resample_mha.h"
#include "getopt.h"

void
print_usage (void)
{
    printf ("Usage: resample_mha [options]\n");
    printf ("Required:   --input=file\n"
	    "            --output=file\n"
	    "            --input_type={uchar,short,ushort,float,vf}\n"
	    "            --output_type={uchar,short,ushort,float,vf}\n"
	    "Optional:   --subsample=\"x y z\"\n"
	    "            --origin=\"x y z\"\n"
	    "            --spacing=\"x y z\"\n"
	    "            --size=\"x y z\"\n"
		"            --interpolation={nn, linear}\n"
	    "            --default_val=val\n");
    exit (1);
}

void
show_stats (ShortImageType::Pointer image)
{
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    const ShortImageType::IndexType& st = region.GetIndex();
    const ShortImageType::SizeType& sz = image->GetLargestPossibleRegion().GetSize();
    const ShortImageType::PointType& og = image->GetOrigin();
    const ShortImageType::SpacingType& sp = image->GetSpacing();

    printf ("Origin = %g %g %g\n", og[0], og[1], og[2]);
    printf ("Spacing = %g %g %g\n", sp[0], sp[1], sp[2]);
    std::cout << "Start = " << st[0] << " " << st[1] << " " << st[2] << std::endl;
    std::cout << "Size = " << sz[0] << " " << sz[1] << " " << sz[2] << std::endl;
}

void
shift_pet_values (FloatImageType::Pointer image)
{
    printf ("Shifting values for pet...\n");
    typedef itk::ImageRegionIterator< FloatImageType > IteratorType;
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	float f = it.Get();
	f = f - 100;
	if (f < 0) {
	    f = f * 10;
	} else if (f > 0x4000) {
	    f = 0x4000 + (f - 0x4000) / 2;
	}
	if (f > 0x07FFF) {
	    f = 0x07FFF;
	}
	it.Set(f);
    }
}

void
fix_invalid_pixels_with_shift (ShortImageType::Pointer image)
{
    typedef itk::ImageRegionIterator< ShortImageType > IteratorType;
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	short c = it.Get();
	if (c < -1000) {
	    c = -1000;
	}
	it.Set (c + 1000);
    }
}

void
fix_invalid_pixels (ShortImageType::Pointer image)
{
    typedef itk::ImageRegionIterator< ShortImageType > IteratorType;
    ShortImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	short c = it.Get();
	if (c < -1000) {
	    it.Set (-1000);
	}
    }
}

void
parse_args (Resample_Parms* resp, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "input_type",     required_argument,      NULL,           1 },
	{ "output_type",    required_argument,      NULL,           2 },
	{ "input",          required_argument,      NULL,           3 },
	{ "output",         required_argument,      NULL,           4 },
	{ "subsample",      required_argument,      NULL,           5 },
	{ "origin",         required_argument,      NULL,           6 },
	{ "spacing",        required_argument,      NULL,           7 },
	{ "size",           required_argument,      NULL,           8 },
	{ "interpolation",  required_argument,      NULL,           9 },
	{ "default_val",    required_argument,      NULL,           10 },
	{ NULL,             0,                      NULL,           0 }
    };

    char **parm_argv = argv;
    int parm_argc = argc;

    while ((ch = getopt_long(parm_argc, parm_argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    if (!strcmp(optarg,"ushort") || !strcmp(optarg,"unsigned")) {
		resp->input_type = PLM_IMG_TYPE_ITK_USHORT;
	    }
	    else if (!strcmp(optarg,"short") || !strcmp(optarg,"signed")) {
		resp->input_type = PLM_IMG_TYPE_ITK_SHORT;
	    }
	    else if (!strcmp(optarg,"float")) {
		resp->input_type = PLM_IMG_TYPE_ITK_FLOAT;
	    }
	    else if (!strcmp(optarg,"mask") || !strcmp(optarg,"uchar")) {
		resp->input_type = PLM_IMG_TYPE_ITK_UCHAR;
	    }
	    else if (!strcmp(optarg,"vf")) {
		resp->input_type = PLM_IMG_TYPE_ITK_FLOAT_FIELD;
	    }
	    else {
		print_usage();
	    }
	    break;
	case 2:
	    if (!strcmp(optarg,"ushort") || !strcmp(optarg,"unsigned")) {
		resp->output_type = PLM_IMG_TYPE_ITK_USHORT;
	    }
	    else if (!strcmp(optarg,"short") || !strcmp(optarg,"signed")) {
		resp->output_type = PLM_IMG_TYPE_ITK_SHORT;
	    }
	    else if (!strcmp(optarg,"float")) {
		resp->output_type = PLM_IMG_TYPE_ITK_FLOAT;
	    }
	    else if (!strcmp(optarg,"mask") || !strcmp(optarg,"uchar")) {
		resp->output_type = PLM_IMG_TYPE_ITK_UCHAR;
	    }
	    else if (!strcmp(optarg,"vf")) {
		resp->output_type = PLM_IMG_TYPE_ITK_FLOAT_FIELD;
	    }
	    else {
		print_usage();
	    }
	    break;
	case 3:
	    strncpy (resp->mha_in_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (resp->mha_out_fn, optarg, _MAX_PATH);
	    break;
	case 5:
	    rc = sscanf (optarg, "%d %d %d", &(resp->subsample[0]), &(resp->subsample[1]), &(resp->subsample[2]));
	    if (rc != 3) {
		printf ("Subsampling option must have three arguments\n");
		exit (1);
	    }
	    resp->have_subsample = 1;
	    break;
	case 6:
	    rc = sscanf (optarg, "%g %g %g", &(resp->origin[0]), &(resp->origin[1]), &(resp->origin[2]));
	    if (rc != 3) {
		printf ("Origin option must have three arguments\n");
		exit (1);
	    }
	    resp->have_origin = 1;
	    break;
	case 7:
	    rc = sscanf (optarg, "%g %g %g", &(resp->spacing[0]), &(resp->spacing[1]), &(resp->spacing[2]));
	    if (rc != 3) {
		printf ("Spacing option must have three arguments\n");
		exit (1);
	    }
	    resp->have_spacing = 1;
	    break;
	case 8:
	    rc = sscanf (optarg, "%d %d %d", &(resp->size[0]), &(resp->size[1]), &(resp->size[2]));
	    if (rc != 3) {
		printf ("Size option must have three arguments\n");
		exit (1);
	    }
	    resp->have_size = 1;
	    break;
	case 9:
	    if (!strcmp (optarg, "nn")) {
		resp->interp_lin = 0;
	    } else if (!strcmp (optarg, "linear")) {
		resp->interp_lin = 1;
	    } else {
		fprintf (stderr, "Error.  --interpolation must be either nn or linear.\n");
		print_usage ();
	    }
	    break;
	case 10:
	    rc = sscanf (optarg, "%g", &(resp->default_val));
	    if (rc != 1) {
		printf ("Default value option must have one arguments\n");
		exit (1);
	    }
	    resp->have_default_val = 1;
	    break;
	default:
	    break;
	}
    }
    if (!resp->mha_in_fn[0] || !resp->mha_out_fn[0]) {
	printf ("Error: must specify --input and --output\n");
	print_usage();
    }
    if (resp->input_type == PLM_IMG_TYPE_UNDEFINED || resp->output_type == PLM_IMG_TYPE_UNDEFINED) {
	printf ("Error: must specify --input_type and --output_type\n");
	print_usage();
    }
}

void
do_resampling (Resample_Parms* resp)
{
    typedef itk::ImageFileWriter < UCharImageType > UCharWriterType;
    typedef itk::ImageFileWriter < ShortImageType > ShortWriterType;
    typedef itk::ImageFileWriter < UShortImageType > UShortWriterType;
    typedef itk::ImageFileWriter < FloatImageType > FloatWriterType;

    if (resp->input_type == PLM_IMG_TYPE_ITK_USHORT) {
	/* Do nothing for now */
	printf ("Unsigned short not yet supported.\n");
    }
    else if (resp->input_type == PLM_IMG_TYPE_ITK_SHORT) {

	ShortImageType::Pointer input_image = load_short (resp->mha_in_fn, 0);

	if (resp->have_subsample) {
	    input_image = subsample_image (input_image, resp->subsample[0],
		    resp->subsample[1], resp->subsample[2], resp->default_val);
	}
	else if (resp->have_origin && resp->have_spacing && resp->have_size) {
	    input_image = resample_image (input_image, resp->origin, resp->spacing, resp->size, resp->default_val, resp->interp_lin);
	}
	if (resp->output_type == PLM_IMG_TYPE_ITK_SHORT) {
	    //just output
	    fix_invalid_pixels (input_image);

	    ShortWriterType::Pointer writer = ShortWriterType::New();
	    writer->SetFileName (resp->mha_out_fn);
	    writer->SetInput (input_image);
	    writer->Update();

	}
	else if (resp->output_type == PLM_IMG_TYPE_ITK_USHORT) {
	    if (resp->adjust == 0) {
		fix_invalid_pixels_with_shift (input_image);
	    }

	    typedef itk::CastImageFilter <ShortImageType,
		    UShortImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();
	    caster->SetInput(input_image);
        
	    UShortWriterType::Pointer writer = UShortWriterType::New();

	    writer->SetFileName (resp->mha_out_fn);
	    writer->SetInput(caster->GetOutput());
	    writer->Update();
	}
	else {
	    /* Do nothing for now */
	    printf ("Error. Unimplemented conversion type.\n");
	    exit (-1);
	}
    }
    else if (resp->input_type == PLM_IMG_TYPE_ITK_FLOAT) {

	FloatImageType::Pointer input_image = load_float (resp->mha_in_fn, 0);

	if (resp->have_subsample) {
	    input_image = subsample_image (input_image, resp->subsample[0],
		    resp->subsample[1], resp->subsample[2], resp->default_val);
	}
	else if (resp->have_origin && resp->have_spacing && resp->have_size) {
	    input_image = resample_image (input_image, resp->origin, resp->spacing, resp->size, resp->default_val, resp->interp_lin);
	}

	if (resp->output_type == PLM_IMG_TYPE_ITK_FLOAT) {
	    FloatWriterType::Pointer writer = FloatWriterType::New();
	    writer->SetFileName (resp->mha_out_fn);
	    writer->SetInput (input_image);
	    writer->Update();
	} else if (resp->output_type == PLM_IMG_TYPE_ITK_SHORT) {

	    typedef itk::CastImageFilter <FloatImageType,
		    ShortImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();
	    caster->SetInput(input_image);
        
	    ShortWriterType::Pointer writer = ShortWriterType::New();

	    writer->SetFileName (resp->mha_out_fn);
	    writer->SetInput(caster->GetOutput());
	    writer->Update();
	} else {
	    /* Do nothing for now */
	    printf ("Error. Unimplemented conversion type.\n");
	    exit (-1);
	}
    }
    else if (resp->input_type == PLM_IMG_TYPE_ITK_UCHAR) {
	UCharImageType::Pointer input_image = load_uchar (resp->mha_in_fn, 0);
	if (resp->have_subsample) {
	    input_image = subsample_image (input_image, resp->subsample[0],
		    resp->subsample[1], resp->subsample[2], resp->default_val);
	}
	else if (resp->have_origin && resp->have_spacing && resp->have_size) {
	    input_image = resample_image (input_image, resp->origin, resp->spacing, resp->size, resp->default_val, resp->interp_lin);
	}

	if (resp->output_type == PLM_IMG_TYPE_ITK_UCHAR) {

	    UCharWriterType::Pointer writer = UCharWriterType::New();
	    writer->SetFileName (resp->mha_out_fn);
	    writer->SetInput(input_image);
	    writer->Update();
	} else if (resp->output_type == PLM_IMG_TYPE_ITK_SHORT) {
	    typedef itk::CastImageFilter <UCharImageType,
		    ShortImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();
	    caster->SetInput(input_image);
        
	    ShortWriterType::Pointer writer = ShortWriterType::New();

	    writer->SetFileName (resp->mha_out_fn);
	    writer->SetInput(caster->GetOutput());
	    writer->Update();
	} else {
	    /* Do nothing for now */
	    printf ("Error. Unimplemented conversion type.\n");
	    exit (-1);
	}
    }
    else if (resp->input_type == PLM_IMG_TYPE_ITK_FLOAT_FIELD) {
	if (resp->output_type == PLM_IMG_TYPE_ITK_FLOAT_FIELD) {
	    DeformationFieldType::Pointer input_field = load_float_field (resp->mha_in_fn);
	    if (resp->have_subsample) {
		/* Do nothing for now */
		printf ("Error. Unimplemented conversion type.\n");
		exit (-1);
	    }
	    else if (resp->have_origin && resp->have_spacing && resp->have_size) {
		printf ("Resampling...\n");
		input_field = vector_resample_image (input_field, resp->origin, resp->spacing, resp->size);
	    }
	    itk_image_save (input_field, resp->mha_out_fn);
	} else {
	    /* Do nothing for now */
	    printf ("Error. Unimplemented conversion type.\n");
	    exit (-1);
	}
    }
}

int
main (int argc, char *argv[])
{
    Resample_Parms resp;
    parse_args (&resp, argc, argv);

    do_resampling (&resp);

    printf ("Finished!\n");
    return 0;
}
