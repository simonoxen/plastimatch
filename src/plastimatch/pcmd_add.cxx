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
