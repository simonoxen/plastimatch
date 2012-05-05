/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"
#include "getopt.h"

#include "plmbase.h"
#include "plmsys.h"

#include "dvh.h"
#include "make_string.h"
#include "plm_image.h"
#include "rtds.h"
#include "rtss.h"
#include "rtss_structure.h"


Dvh_parms::Dvh_parms ()
{
    input_units = DVH_UNITS_GY;
    normalization = DVH_NORMALIZATION_PCT;
    cumulative = 0;
    printf ("Initializing num_bins to 256\n");
    num_bins = 256;
    bin_width = 1;
}

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

std::string
dvh_execute_internal (
    Rtds *rtds,
    Dvh_parms *parms)
{
    int *hist;
    int *struct_vox;
    int bin;
    float dose_spacing[3];
    float ss_spacing[3];
    float ss_ori[3];
    plm_long ss_dim[3];
    std::string output_string = "";

    FloatImageType::Pointer dose_img = rtds->m_dose->itk_float ();
    UInt32ImageType::Pointer ss_img = rtds->m_rtss->get_ss_img();

    /* GCS HACK: This should go into rtss.cxx */
    Rtss_polyline_set *ss_list;
    if (rtds->m_rtss->m_ss_list) {
        ss_list = rtds->m_rtss->get_ss_list();
    } else {
        if (rtds->m_rtss->m_cxt) {
            ss_list = rtds->m_rtss->m_cxt;
        } else {
            ss_list = new Rtss_polyline_set;
            for (int i = 0; i < 32; i++) {
                Rtss_structure* structure = 
                    ss_list->add_structure ("Unknown Structure", 
                        "255 255 0", i);
                structure->bit = i;
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
    dose_spacing[0]=dose_img->GetSpacing()[0];
    dose_spacing[1]=dose_img->GetSpacing()[1];
    dose_spacing[2]=dose_img->GetSpacing()[2];
    std::cout << dose_spacing[0] << " " 
        << dose_spacing[1] << " " << dose_spacing[2] << "\n";
    ss_spacing[0]=ss_img->GetSpacing()[0];
    ss_spacing[1]=ss_img->GetSpacing()[1];
    ss_spacing[2]=ss_img->GetSpacing()[2];
    std::cout << ss_spacing[0] << " " 
        << ss_spacing[1] << " " << ss_spacing[2] << "\n";

    if (dose_spacing[0] != ss_spacing[0] 
        || dose_spacing[1] != ss_spacing[1] 
        || dose_spacing[2] != ss_spacing[2])
    {
        std::cout << "dose voxel " 
            << dose_spacing[0] << " " 
            << dose_spacing[1] << " " 
            << dose_spacing[2] << std::endl;
        std::cout << "ss voxel " 
            << ss_spacing[0] << " " 
            << ss_spacing[1] << " " 
            << ss_spacing[2] << std::endl;
        std::cout << "Resampling" << std::endl;

        /*resample volume*/
        ss_ori[0]=ss_img->GetOrigin()[0];
        ss_ori[1]=ss_img->GetOrigin()[1];
        ss_ori[2]=ss_img->GetOrigin()[2];
        ss_dim[0]=ss_img->GetLargestPossibleRegion().GetSize()[0];
        ss_dim[1]=ss_img->GetLargestPossibleRegion().GetSize()[1];
        ss_dim[2]=ss_img->GetLargestPossibleRegion().GetSize()[2];

        /* GCS FIX: Direction cosines */
        Plm_image_header pih (ss_dim, ss_ori, ss_spacing, 0);
        FloatImageType::Pointer resampled 
            = resample_image (dose_img, &pih, 0, 1);
        dose_img=resampled;

    } else {
        std::cout 
            << "Dose and ss-img have the same size. Resample not necessary.\n";
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
        uint32_t s = it_s.Get();

        /* Convert from Gy to cGy */
        if (parms->input_units == DVH_UNITS_GY) {
            d = d * 100;
        }

        /* Compute the bin */
        bin = (int) floor ((d+(0.5*parms->bin_width)) / parms->bin_width);
        if (bin < 0) {
            bin = 0;
        } else if (bin > (parms->num_bins-1)) {
            bin = parms->num_bins - 1;
        }

        for (size_t sno = 0; sno < ss_list->num_structures; sno++) {
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
        for (size_t sno = 0; sno < ss_list->num_structures; sno++) {
            int cum = 0;
            for (bin = parms->num_bins - 1; bin >= 0; bin--) {
                cum = cum + hist[bin*ss_list->num_structures + sno];
                hist[bin*ss_list->num_structures + sno] = cum;
            }
        }
    }

    /* Create output string */
    output_string = "Dose (cGy)";
    for (size_t sno = 0; sno < ss_list->num_structures; sno++) {
        Rtss_structure *curr_structure = ss_list->slist[sno];
        output_string += ",";
        output_string += (const char*) curr_structure->name;
    }
    output_string += "\n";
    for (plm_long bin = 0; bin < parms->num_bins; bin++) {
        output_string += make_string (bin * parms->bin_width);
        for (size_t sno = 0; sno < ss_list->num_structures; sno++) {
            int val = hist[bin*ss_list->num_structures + sno];
            output_string += ",";
            if (struct_vox[sno] == 0) {
                output_string += "0";
            }
            else if (parms->normalization == DVH_NORMALIZATION_PCT) {
                float fval = ((float) val) / struct_vox[sno];
                output_string += make_string (fval);
            }
            else {
                output_string += make_string (val);
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
    rtds.m_rtss = new Rtss (&rtds);
    rtds.m_rtss->m_ss_img = input_ss_img;
    output_string = dvh_execute_internal (&rtds, parms);

    /* Rtds will delete these in the destructor, but we want 
       to let the caller delete */
    rtds.m_dose = 0;
    rtds.m_rtss->m_ss_img = 0;

    return output_string;
}

std::string
dvh_execute (
    UInt32ImageType::Pointer input_ss_img,
    FloatImageType::Pointer input_dose_img,
    Dvh_parms *parms)
{
    Rtds rtds;
    std::string output_string;

    printf ("Hello from slicer dvh plugin!  :)\n");

    rtds.m_dose = new Plm_image;
    rtds.m_dose->set_itk (input_dose_img);

    Plm_image *plm_ss_img = new Plm_image;
    plm_ss_img->set_itk (input_ss_img);

    rtds.m_rtss = new Rtss (&rtds);
    rtds.m_rtss->m_ss_img = plm_ss_img;

    output_string = dvh_execute_internal (&rtds, parms);

    /* Ok to let destructor destroy things */

    return output_string;
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
