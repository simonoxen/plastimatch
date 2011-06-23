/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "getopt.h"

#include "bstring_util.h"
#include "dvh.h"
#include "file_util.h"
#include "itk_image.h"
#include "pcmd_dvh.h"
#include "plm_image.h"
#include "plm_int.h"
#include "rtds.h"
#include "rtss.h"
#include "rtss_structure.h"

static void
load_input_files (Rtds *rtds, Dvh_parms_pcmd *parms)
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

#if defined (commentout)
void
dvh_execute (
    Dvh_parms* parms)
{
    int sno;
    Rtds rtds;
    int *hist;
    int *struct_vox;
    int bin;
    float dose_vox[3];
    float ss_vox[3];
    float ss_ori[3];
    int ss_size[3];

    /* Load input files */
    std::cout << "Loading..." << std::endl;
    load_input_files (&rtds, parms);
    FloatImageType::Pointer dose_img = rtds.m_dose->itk_float ();
    UInt32ImageType::Pointer ss_img = rtds.m_ss_image->get_ss_img();
    Rtss_polyline_set *ss_list = rtds.m_ss_image->get_ss_list();

    /* Create histogram */
    std::cout << "Creating Histogram..." << std::endl;
    printf ("Your Histogram will have %d bins and will be %f cGy large\n",
	parms->num_bins, parms->bin_width);
    hist = (int*) malloc (sizeof(int) * ss_list->num_structures 
	* parms->num_bins);
    memset (hist, 0, sizeof(int) * ss_list->num_structures 
	* parms->num_bins);
    struct_vox = (int*) malloc (sizeof(int) * ss_list->num_structures);
    memset (struct_vox, 0, sizeof(int) * ss_list->num_structures);
   
    /* Is voxel size the same? */
    std::cout << "checking voxel size..." << std::endl;
    dose_vox[0]=dose_img->GetSpacing()[0];
    dose_vox[1]=dose_img->GetSpacing()[1];
    dose_vox[2]=dose_img->GetSpacing()[2];
    ss_vox[0]=ss_img->GetSpacing()[0];
    ss_vox[1]=ss_img->GetSpacing()[1];
    ss_vox[2]=ss_img->GetSpacing()[2];

    if (dose_vox[0] != ss_vox[0] 
	|| dose_vox[1] != ss_vox[1] 
	|| dose_vox[2] != ss_vox[2])
    {
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

	for (sno = 0; sno < ss_list->num_structures; sno++) {
	    Rtss_structure *curr_structure = ss_list->slist[sno];
		    
	    /* Is this pixel in the current structure? */
	    uint32_t in_struct = s & (1 << curr_structure->bit);

	    /* If so, update histogram & structure size */
	    if (in_struct) {
		struct_vox[sno] ++;
		hist[bin*ss_list->num_structures + sno] ++;
	    }
	}
    }

    /* Convert histogram to cumulative histogram */
    if (parms->cumulative) {
	for (sno = 0; sno < ss_list->num_structures; sno++) {
	    int cum = 0;
	    for (bin = parms->num_bins - 1; bin >= 0; bin--) {
		cum = cum + hist[bin*ss_list->num_structures + sno];
		hist[bin*ss_list->num_structures + sno] = cum;
	    }
	}
    }

    /* Save the csv file */
    FILE *fp = fopen (parms->output_csv_fn, "w");
    fprintf (fp, "Dose (cGy)");
    for (sno = 0; sno < ss_list->num_structures; sno++) {
	Rtss_structure *curr_structure = ss_list->slist[sno];
	fprintf (fp, ",%s", (const char*) curr_structure->name);
    }
    fprintf (fp, "\n");
    for (bin = 0; bin < parms->num_bins; bin++) {
	fprintf (fp, "%g", bin * parms->bin_width);
	for (sno = 0; sno < ss_list->num_structures; sno++) {
	    int val = hist[bin*ss_list->num_structures + sno];
	    if (parms->normalization == DVH_NORMALIZATION_PCT) {
		float fval = ((float) val) / struct_vox[sno];
		fprintf (fp, ",%f", fval);
	    } else {
		fprintf (fp, ",%d", val);
	    }
	}
	fprintf (fp, "\n");
    }
    fclose (fp);
}
#endif

std::string
dvh_execute_internal (
    Rtds *rtds,
    Dvh_parms *parms)
{
    int sno;
    int *hist;
    int *struct_vox;
    int bin;
    float dose_vox[3];
    float ss_vox[3];
    float ss_ori[3];
    int ss_size[3];
    std::string output_string = "";

    FloatImageType::Pointer dose_img = rtds->m_dose->itk_float ();
    UInt32ImageType::Pointer ss_img = rtds->m_ss_image->get_ss_img();

    /* GCS HACK: This should go into rtss.cxx */
    Rtss_polyline_set *ss_list;
    if (rtds->m_ss_image->m_ss_list) {
	ss_list = rtds->m_ss_image->get_ss_list();
    } else {
	if (rtds->m_ss_image->m_cxt) {
	    ss_list = rtds->m_ss_image->m_cxt;
	} else {
	    ss_list = new Rtss_polyline_set;
	    for (int i = 0; i < 32; i++) {
		ss_list->add_structure ("Unknown Structure", 
		    "255 255 0", i);
	    }
	}
    }

    /* Create histogram */
    std::cout << "Creating Histogram..." << std::endl;
    printf ("Your Histogram will have %d bins and will be %f cGy large\n",
	parms->num_bins, parms->bin_width);
    hist = (int*) malloc (sizeof(int) * ss_list->num_structures 
	* parms->num_bins);
    memset (hist, 0, sizeof(int) * ss_list->num_structures 
	* parms->num_bins);
    struct_vox = (int*) malloc (sizeof(int) * ss_list->num_structures);
    memset (struct_vox, 0, sizeof(int) * ss_list->num_structures);
   
    /* Is voxel size the same? */
    std::cout << "checking voxel size..." << std::endl;
    dose_vox[0]=dose_img->GetSpacing()[0];
    dose_vox[1]=dose_img->GetSpacing()[1];
    dose_vox[2]=dose_img->GetSpacing()[2];
    std::cout << dose_vox[0] << " " 
	<< dose_vox[1] << " " << dose_vox[2] << "\n";
    ss_vox[0]=ss_img->GetSpacing()[0];
    ss_vox[1]=ss_img->GetSpacing()[1];
    ss_vox[2]=ss_img->GetSpacing()[2];
    std::cout << ss_vox[0] << " " 
	<< ss_vox[1] << " " << ss_vox[2] << "\n";

    if (dose_vox[0] != ss_vox[0] 
	|| dose_vox[1] != ss_vox[1] 
	|| dose_vox[2] != ss_vox[2])
    {
	std::cout << "dose voxel " 
	    << dose_vox[0] << " " 
	    << dose_vox[1] << " " 
	    << dose_vox[2] << std::endl;
	std::cout << "ss voxel " 
	    << ss_vox[0] << " " 
	    << ss_vox[1] << " " 
	    << ss_vox[2] << std::endl;
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

	for (sno = 0; sno < ss_list->num_structures; sno++) {
	    Rtss_structure *curr_structure = ss_list->slist[sno];
		    
	    /* Is this pixel in the current structure? */
	    uint32_t in_struct = s & (1 << curr_structure->bit);

	    /* If so, update histogram & structure size */
	    if (in_struct) {
		struct_vox[sno] ++;
		hist[bin*ss_list->num_structures + sno] ++;
	    }
	}
    }

    /* Convert histogram to cumulative histogram */
    if (parms->cumulative) {
	for (sno = 0; sno < ss_list->num_structures; sno++) {
	    int cum = 0;
	    for (bin = parms->num_bins - 1; bin >= 0; bin--) {
		cum = cum + hist[bin*ss_list->num_structures + sno];
		hist[bin*ss_list->num_structures + sno] = cum;
	    }
	}
    }

    /* Create output string */
    output_string = "Dose (cGy)";
    for (sno = 0; sno < ss_list->num_structures; sno++) {
	Rtss_structure *curr_structure = ss_list->slist[sno];
	output_string += ",";
	output_string += (const char*) curr_structure->name;
    }
    output_string += "\n";
    for (bin = 0; bin < parms->num_bins; bin++) {
	output_string += bin * parms->bin_width;
	for (sno = 0; sno < ss_list->num_structures; sno++) {
	    int val = hist[bin*ss_list->num_structures + sno];
	    if (parms->normalization == DVH_NORMALIZATION_PCT) {
		float fval = ((float) val) / struct_vox[sno];
		output_string += ",";
		output_string += fval;
	    } else {
		output_string += ",";
		output_string += val;
	    }
	}
	output_string += "\n";
    }
    return output_string;
}

std::string
dvh_execute (
    Plm_image *input_ss_img,
    Plm_image *input_dose_img,
    Dvh_parms *parms)
{
    Rtds rtds;
    std::string output_string;

    rtds.m_dose = input_dose_img;
    rtds.m_ss_image = new Rtss (&rtds);
    rtds.m_ss_image->m_ss_img = input_ss_img;
    output_string = dvh_execute_internal (&rtds, parms);

    /* Rtds will delete these in the destructor, but we want 
       to let the caller delete */
    rtds.m_dose = 0;
    rtds.m_ss_image->m_ss_img = 0;

    return 0;
}

std::string
dvh_execute (
    UShortImageType::Pointer input_ss_img,
    UShortImageType::Pointer input_dose_img,
    Dvh_parms *parms)
{
    Rtds rtds;
    std::string output_string;

    return 0;
}

void
dvh_execute (
    Dvh_parms_pcmd *parms)
{
    Rtds rtds;
    std::string output_string;

    /* Load input files */
    std::cout << "Loading..." << std::endl;
    load_input_files (&rtds, parms);

    output_string = dvh_execute_internal (&rtds, &parms->dvh_parms);

    make_directory_recursive (parms->output_csv_fn);
    FILE *fp = fopen (parms->output_csv_fn, "w");
    fprintf (fp, "%s", output_string.c_str());
    fclose (fp);
}
