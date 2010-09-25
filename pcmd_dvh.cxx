/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "getopt.h"

#include "bstring_util.h"
#include "itk_image.h"
#include "pcmd_dvh.h"
#include "plm_image.h"
#include "plm_int.h"
#include "rtds.h"

static void
load_input_files (Rtds *rtds, Dvh_parms *parms)
{
    if (parms->input_ss_img_fn.length() != 0) {
	rtds->load_ss_img (
	    (const char*) parms->input_ss_img_fn, 
	    (const char*) parms->input_ss_list_fn);
    }
    if (parms->input_dose_fn.length() != 0) {
	rtds->m_dose = plm_image_load_native (
	    (const char*) parms->input_dose_fn);
    }
}

static void
dvh_main (Dvh_parms* parms)
{
    int sno;
    Rtds rtds;
    int *hist;
    int bin;
    float dose_vox[3];
    float ss_vox[3];
    float ss_ori[3];
    int ss_size[3];

    /* Load input files */
    std::cout << "Loading..." << std::endl;
    load_input_files (&rtds, parms);
    FloatImageType::Pointer dose_img = rtds.m_dose->itk_float ();
    UInt32ImageType::Pointer ss_img = rtds.m_ss_img->m_itk_uint32;

    /* Create histogram */
    std::cout << "Creating Histogram..." << std::endl;
    std::cout << "Your Histogram will have " << parms->num_bins << " bins and they will be " << parms->bin_width << "Gy large" << std::endl;
    hist = (int*) malloc (sizeof(int) * rtds.m_ss_list->num_structures 
	* parms->num_bins);
    memset (hist, 0, sizeof(int) * rtds.m_ss_list->num_structures 
	* parms->num_bins);

    /* Is voxel size the same? */
    std::cout << "checking voxel size..." << std::endl;
    dose_vox[0]=dose_img->GetSpacing()[0];
    dose_vox[1]=dose_img->GetSpacing()[1];
    dose_vox[2]=dose_img->GetSpacing()[2];
    ss_vox[0]=ss_img->GetSpacing()[0];
    ss_vox[1]=ss_img->GetSpacing()[1];
    ss_vox[2]=ss_img->GetSpacing()[2];

    if (dose_vox[0]!=ss_vox[0] || dose_vox[1]!=ss_vox[1] || dose_vox[2]!=ss_vox[2]) {
	std::cout << "dose voxel " << dose_vox[0] <<" "<< dose_vox[1] <<" "<< dose_vox[2] << std::endl;
	std::cout << "ss voxel " << ss_vox[0] <<" "<< ss_vox[1] <<" "<< ss_vox[2] << std::endl;
	std::cout << "Resampling" << std::endl;

	/*resample volume*/
	ss_ori[0]=ss_img->GetOrigin()[0];
	ss_ori[1]=ss_img->GetOrigin()[1];
	ss_ori[2]=ss_img->GetOrigin()[2];
	ss_size[0]=ss_img->GetLargestPossibleRegion().GetSize()[0];
	ss_size[1]=ss_img->GetLargestPossibleRegion().GetSize()[1];
	ss_size[2]=ss_img->GetLargestPossibleRegion().GetSize()[2];
	FloatImageType::Pointer resampled 
	    = resample_image (dose_img, ss_ori, ss_vox, ss_size , 0, 1);
	dose_img=resampled;
	/*create correct volume*/

    } else {
	std::cout << "dose and ss-img have the same size...continue computation" <<std::endl;
    }

    /* Declare iterators */
    typedef itk::ImageRegionConstIterator < FloatImageType > 
	FloatIteratorType;
    typedef itk::ImageRegionConstIterator < UInt32ImageType > 
	UInt32IteratorType;
    FloatIteratorType it_d (dose_img, dose_img->GetRequestedRegion ());
    UInt32IteratorType it_s (ss_img, ss_img->GetRequestedRegion ());

    /* Loop through dose & ss images */
    for (it_d.GoToBegin(), it_s.GoToBegin(); 
	 !it_d.IsAtEnd(); 
	 ++it_d, ++it_s)
    {
	float d = it_d.Get();

	/* Convert from Gy to cGy */
	if (parms->input_units == DVH_UNITS_GY) {
	    d = d * 100;
	}
	uint32_t s = it_s.Get();

	/* Compute the bin */
	bin = (int) floor ((d+(0.5*parms->bin_width)) / parms->bin_width);
	if (bin < 0) {
	    bin = 0;
	} else if (bin > (parms->num_bins-1)) {
	    bin = parms->num_bins - 1;
	}

	for (sno = 0; sno < rtds.m_ss_list->num_structures; sno++) {
	    Rtss_structure *curr_structure = rtds.m_ss_list->slist[sno];
		    
	    /* Is this pixel in the current structure? */
	    uint32_t in_struct = s & (1 << curr_structure->bit);

	    /* If so, update histogram */
	    if (in_struct) {
		hist[bin*rtds.m_ss_list->num_structures + sno] ++;
	    }
	}
    }

    /* Convert histogram to cumulative histogram */
    if (parms->cumulative) {
	for (sno = 0; sno < rtds.m_ss_list->num_structures; sno++) {
	    int cum = 0;
	    for (bin = parms->num_bins - 1; bin >= 0; bin--) {
		cum = cum + hist[bin*rtds.m_ss_list->num_structures + sno];
		hist[bin*rtds.m_ss_list->num_structures + sno] = cum;
	    }
	}
    }


    /* Save the csv file */
    FILE *fp = fopen (parms->output_csv_fn, "w");
    fprintf (fp, "Dose (cGy)");
    for (sno = 0; sno < rtds.m_ss_list->num_structures; sno++) {
	Rtss_structure *curr_structure = rtds.m_ss_list->slist[sno];
	fprintf (fp, ",%s", (const char*) curr_structure->name);
    }
    fprintf (fp, "\n");
    for (bin = 0; bin < parms->num_bins; bin++) {
	fprintf (fp, "%g", bin * parms->bin_width);
	for (sno = 0; sno < rtds.m_ss_list->num_structures; sno++) {
	    int val = hist[bin*rtds.m_ss_list->num_structures + sno];
	    fprintf (fp, ",%d", val);
	}
	fprintf (fp, "\n");
    }
    fclose (fp);
}

static void
print_usage (void)
{
    printf (
	"Usage: plastimatch dvh [options]\n"
	"   --input-ss-img file\n"
	"   --input-ss-list file\n"
	"   --input-dose file\n"
	"   --output-csv file\n"
	"   --input-uints {gy,cgy}\n"
	"   --cumulative\n"
	"   --num-bins\n"
	"   --bin-width\n"
    );
    exit (-1);
}

static void
parse_args (Dvh_parms* parms, int argc, char* argv[])
{
    int rc;
    int ch;
    static struct option longopts[] = {
	{ "input_ss_img",   required_argument,      NULL,           2 },
	{ "input-ss-img",   required_argument,      NULL,           2 },
	{ "input_ss_list",  required_argument,      NULL,           3 },
	{ "input-ss-list",  required_argument,      NULL,           3 },
	{ "input_dose",     required_argument,      NULL,           4 },
	{ "input-dose",     required_argument,      NULL,           4 },
	{ "output_csv",     required_argument,      NULL,           5 },
	{ "output-csv",     required_argument,      NULL,           5 },
	{ "input_units",    required_argument,      NULL,           6 },
	{ "input-units",    required_argument,      NULL,           6 },
	{ "cumulative",     no_argument,            NULL,           7 },
	{ "num_bins",       required_argument,      NULL,           8 },
	{ "num-bins",       required_argument,      NULL,           8 },
	{ "bin_width",      required_argument,      NULL,           9 },
	{ "bin-width",      required_argument,      NULL,           9 },
	{ NULL,             0,                      NULL,           0 }
    };

    /* Skip command "dvh" */
    optind ++;

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    parms->input_ss_img_fn = optarg;
	    break;
	case 3:
	    parms->input_ss_list_fn = optarg;
	    break;
	case 4:
	    parms->input_dose_fn = optarg;
	    break;
	case 5:
	    parms->output_csv_fn = optarg;
	    break;
	case 6:
	    if (!strcmp (optarg, "cgy") || !strcmp (optarg, "cGy"))
	    {
		parms->input_units = DVH_UNITS_CGY;
	    }
	    else if (!strcmp (optarg, "gy") || !strcmp (optarg, "Gy"))
	    {
		parms->input_units = DVH_UNITS_CGY;
	    }
	    else {
		fprintf (stderr, "Error.  Units must be Gy or cGy.\n");
		print_usage ();
	    }
	    break;
	case 7:
	    parms->cumulative = 1;
	    break;
	case 8:
	    rc = sscanf (optarg, "%d", &parms->num_bins);
	    std::cout << "num_bins " << parms->num_bins << std::endl;
	    break;
	case 9:
	    rc = sscanf (optarg, "%f", &parms->bin_width);
	    std::cout << "bin_width " << parms->bin_width << std::endl;
	    break;
	default:
	    fprintf (stderr, "Error.  Unknown option.\n");
	    print_usage ();
	    break;
	}
    }
    if (bstring_empty (parms->input_ss_img_fn)
	|| bstring_empty (parms->input_ss_list_fn)
	|| bstring_empty (parms->input_dose_fn)
	|| bstring_empty (parms->output_csv_fn))
    {
	fprintf (stderr, 
	    "Error.  Must specify input for dose, ss_img, and output file.\n");
	print_usage ();
    }
}

void
do_command_dvh (int argc, char *argv[])
{
    Dvh_parms parms;
    
    parse_args (&parms, argc, argv);

    dvh_main (&parms);
}
