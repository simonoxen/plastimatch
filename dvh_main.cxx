/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdint.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "getopt.h"
#include "dvh_main.h"
#include "itk_image.h"
#include "plm_image.h"
#include "rtds.h"


static void
load_input_files (Rtds *rtds, Dvh_parms *parms)
{
    if (parms->input_ss_img[0]) {
	rtds->load_ss_img (parms->input_ss_img, parms->input_ss_list);
    }
    if (parms->input_dose[0]) {
	rtds->m_dose = plm_image_load_native (parms->input_dose);
    }
}

static void
dvh_main (Dvh_parms* parms)
{
    int sno;
    Rtds rtds;
    int *hist;
    int bin;
    const int num_bins = 200;
    const float bin_width = 50;

    /* Load input files */
    load_input_files (&rtds, parms);
    FloatImageType::Pointer dose = rtds.m_dose->itk_float ();
    UInt32ImageType::Pointer ss = rtds.m_ss_img->m_itk_uint32;

    /* Create histogram */
    hist = (int*) malloc (sizeof(int) * rtds.m_ss_list->num_structures 
	* num_bins);
    memset (hist, 0, sizeof(int) * rtds.m_ss_list->num_structures 
	* num_bins);

    /* Declare iterators */
    typedef itk::ImageRegionConstIterator < FloatImageType > 
	FloatIteratorType;
    typedef itk::ImageRegionConstIterator < UInt32ImageType > 
	UInt32IteratorType;
    FloatIteratorType it_d (dose, dose->GetRequestedRegion ());
    UInt32IteratorType it_s (ss, ss->GetRequestedRegion ());

    /* Loop through dose & ss images */
    for (it_d.GoToBegin(), it_s.GoToBegin(); 
	 !it_d.IsAtEnd(); 
	 ++it_d, ++it_s)
    {
	float d = it_d.Get();
	uint32_t s = it_s.Get();
	int sno;

	/* Compute the bin */
	bin = (int) floor ((d+(0.5*bin_width)) / bin_width);
	if (bin < 0) {
	    bin = 0;
	} else if (bin > (num_bins-1)) {
	    bin = num_bins - 1;
	}


	for (sno = 0; sno < rtds.m_ss_list->num_structures; sno++) {
	    Cxt_structure *curr_structure = &rtds.m_ss_list->slist[sno];
	    
	    /* Is this pixel in the current structure? */
	    uint32_t in_struct = s & (1 << curr_structure->bit);

	    /* If so, update histogram */
	    if (in_struct) {
		hist[bin*rtds.m_ss_list->num_structures + sno] ++;
	    }
	}
    }

    /* Save the csv file */
    FILE *fp = fopen (parms->output_csv, "w");
    for (sno = 0; sno < rtds.m_ss_list->num_structures; sno++) {
	fprintf (fp, "Dose (cGy)");
	Cxt_structure *curr_structure = &rtds.m_ss_list->slist[sno];
	fprintf (fp, ",%s", curr_structure->name);
    }
    fprintf (fp, "\n");
    for (bin = 0; bin < num_bins; bin++) {
	fprintf (fp, "%g", bin * bin_width);
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
    );
    exit (-1);
}

static void
parse_args (Dvh_parms* parms, int argc, char* argv[])
{
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
	{ NULL,             0,                      NULL,           0 }
    };

    /* Skip command "dvh" */
    optind ++;

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->input_ss_img, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->input_ss_list, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (parms->input_dose, optarg, _MAX_PATH);
	    break;
	case 5:
	    strncpy (parms->output_csv, optarg, _MAX_PATH);
	    break;
	default:
	    fprintf (stderr, "Error.  Unknown option.\n");
	    print_usage ();
	    break;
	}
    }
    if (!parms->input_ss_img[0] || !parms->input_ss_list[0] 
	|| !parms->input_dose[0] || !parms->output_csv[0])
    {
	fprintf (stderr, "Error.  Must specify all inputs.\n");
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
