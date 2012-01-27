/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkAddImageFilter.h"
#include "itkNaryAddImageFilter.h"
#include "itkDivideByConstantImageFilter.h"
#include "getopt.h"
#include "itk_image.h"
#include "itk_image_load.h"
#include "plm_image.h"
#include "plm_path.h"

class Add_parms {
public:
    
};


static void
print_usage (void)
{
    printf (
	//	"Usage: plastimatch add [options]"
	"Usage: plastimatch add"
	" input_file [input_file ...] output_file\n"
    );
    exit (1);
}

void
add_main (int argc, char *argv[])
{
    int i;
    typedef itk::AddImageFilter< FloatImageType, FloatImageType, 
	FloatImageType > AddFilterType;
    typedef itk::DivideByConstantImageFilter< FloatImageType, int, 
	FloatImageType > DivFilterType;

    FloatImageType::Pointer tmp;

    AddFilterType::Pointer addition = AddFilterType::New();
    DivFilterType::Pointer division = DivFilterType::New();

    /* Load the first input image */
    Plm_image *sum = plm_image_load (argv[2], PLM_IMG_TYPE_ITK_FLOAT);

    /* Load and add remaining input images */
    for (i = 3; i < argc - 1; i++) {
	tmp = itk_image_load_float (argv[i], 0);
	addition->SetInput1 (sum->m_itk_float);
	addition->SetInput2 (tmp);
	addition->Update();
	sum->m_itk_float = addition->GetOutput ();
    }

    /* Save the sum image */
    sum->convert_to_original_type ();
    sum->save_image (argv[argc-1]);

#if defined (commentout)
    // divide by the total number of input images
    division->SetConstant(nImages);
    division->SetInput (sumImg);
    division->Update();
    // store the mean image in tmp first before write out
    tmp = division->GetOutput();

    // write the computed mean image
    if (is_directory(outFile)) 
    {
	std::cout << "output dicom to " << outFile << std::endl;
	// Dicom
	itk_image_save_short_dicom (tmp, outFile);
    }
    else
    {
	std::cout << "output to " << outFile << std::endl;
	itk_image_save_short (tmp, outFile);
    }
#endif
}

void
do_command_add (int argc, char *argv[])
{
    if (argc < 4) {
	print_usage ();
    }

    add_main (argc, argv);
}

#if defined (commentout)
static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options]\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Add_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files */
    parser->add_long_option ("", "input", 
	"input directory or filename; "
	"can be an image or vector field", 1, "");
    
    /* Output files */
    parser->add_long_option ("", "output", 
	"output image or vector field", 1, "");

    /* Output options */
    parser->add_long_option ("", "output-type", 
	"type of output image, one of {uchar, short, float, ...}", 1, "");

    /* Algorithm options */
    parser->add_long_option ("", "default-value", 
	"value to set for pixels with unknown value, default is 0", 1, "");
    parser->add_long_option ("", "interpolation", 
	"interpolation type, either \"nn\" or \"linear\", "
	"default is linear", 1, "linear");

    /* Geometry options */
    parser->add_long_option ("F", "fixed", 
	"fixed image (match output size to this image)", 1, "");
    parser->add_long_option ("", "origin", 
	"location of first image voxel in mm \"x y z\"", 1, "");
    parser->add_long_option ("", "dim", 
	"size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "spacing", 
	"voxel spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "direction-cosines", 
	"oriention of x, y, and z axes; Specify either preset value,"
	" {identity,rotated-{1,2,3},sheared},"
	" or 9 digit matrix string \"a b c d e f g h i\"", 1, "");
    parser->add_long_option ("", "subsample", 
	"bin voxels together at integer subsampling rate \"x [y z]\"", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an input file was given */
    if (!parser->option ("input"))
    {
	throw (dlib::error ("Error.  Please specify an input file "
		"using the --input option"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output"))
    {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Check that no extraneous options were given */
    if (parser->number_of_arguments() != 0) {
	std::string extra_arg = (*parser)[0];
	throw (dlib::error ("Error.  Unknown option " + extra_arg));
    }

    /* Input files */
    parms->input_fn = parser->get_string("input").c_str();

    /* Output files */
    parms->output_fn = parser->get_string("output").c_str();

    /* Output options */
    if (parser->option("output-type")) {
	std::string arg = parser->get_string ("output-type");
	parms->output_type = plm_image_type_parse (arg.c_str());
	if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
	    throw (dlib::error ("Error. Unknown --output-type argument: " 
		    + parser->get_string("output-type")));
	}
    }

    /* Algorithm options */
    if (parser->option("default-value")) {
	parms->default_val = parser->get_float("default-value");
    }
    std::string arg = parser->get_string ("interpolation");
    if (arg == "nn") {
	parms->interp_lin = 0;
    }
    else if (arg == "linear") {
	parms->interp_lin = 1;
    }
    else {
	throw (dlib::error ("Error. Unknown --interpolation argument: " 
		+ arg));
    }

    /* Geometry options */
    if (parser->option ("dim")) {
	parms->m_have_dim = 1;
	parser->assign_size_t_13 (parms->dim, "dim");
    }
    if (parser->option ("origin")) {
	parms->m_have_origin = 1;
	parser->assign_float13 (parms->origin, "origin");
    }
    if (parser->option ("spacing")) {
	parms->m_have_spacing = 1;
	parser->assign_float13 (parms->spacing, "spacing");
    }
    if (parser->option ("subsample")) {
	parms->m_have_subsample = 1;
	parser->assign_int13 (parms->subsample, "subsample");
    }
    /* Direction cosines */
    if (parser->option ("direction-cosines")) {
	parms->m_have_direction_cosines = true;
	std::string arg = parser->get_string("direction-cosines");
	if (!parms->m_dc.set_from_string (arg)) {
	    throw (dlib::error ("Error parsing --direction-cosines "
		    "(should have nine numbers)\n"));
	}
    }

    parms->fixed_fn = parser->get_string("fixed").c_str();
}

void
do_command_add (int argc, char *argv[])
{
    Add_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    add_main (&parms);
}
#endif
