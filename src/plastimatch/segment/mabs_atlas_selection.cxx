/* -----------------------------------------------------------------------
 *    See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
 *       ----------------------------------------------------------------------- */
#include "plmsegment_config.h"

#include <stdlib.h>
#include <iterator>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include <itkNormalizedMutualInformationHistogramImageToImageMetric.h>
#include <itkTranslationTransform.h>
#include "itkLinearInterpolateImageFunction.h"

#include "dir_list.h"
#include "logfile.h"
#include "mabs.h"
#include "mabs_atlas_selection.h"
#include "path_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_stages.h"
#include "plm_warp.h"
#include "string_util.h"
#include "registration_parms.h"
#include "registration_data.h"
#include "xform.h"


Mabs_atlas_selection::Mabs_atlas_selection (std::list<std::string> atlases_list, Rt_study::Pointer subject)
{
    //constructor
    this->atlas_dir_list = atlases_list;
    this->subject_rtds = subject;
    this->number_of_atlases = (int) this->atlas_dir_list.size();
}


Mabs_atlas_selection::~Mabs_atlas_selection ()
{
    //destructor
    //delete &atlas_dir_list;
    //delete &subject_rtds;
    delete &number_of_atlases;
}


std::list<std::string>
Mabs_atlas_selection::nmi_ranking(std::string patient_id, const Mabs_parms* parms)
{

    lprintf("NMI RANKING \n");
	
    printf ("Number of initial atlases = %d \n", number_of_atlases);
    double* similarity_value_vector = new double[number_of_atlases];
    
    MaskTypePointer mask;
	
    if (parms->roi_mask_fn.compare("")!=0)
    {
       	Plm_image* mask_plm = plm_image_load (parms->roi_mask_fn, PLM_IMG_TYPE_ITK_FLOAT);
    	
       	mask = MaskType::New();
       	mask->SetImage(mask_plm->itk_uchar());
       	mask->Update();
    }
	
    // Loop through images in the atlas
    std::list<std::string>::iterator atl_it;
    int i;
    for (atl_it = this->atlas_dir_list.begin(), i=0;
         atl_it != this->atlas_dir_list.end(); atl_it++, i++)
    {	
        Rt_study rtds_atl;
       	std::string path_atl = *atl_it;
        std::string atlas_id = basename (path_atl);
        std::string atlas_input_path = string_format ("%s/prealign/%s", "training-mgh", atlas_id.c_str());
        std::string fn = string_format ("%s/img.nrrd", atlas_input_path.c_str());
        rtds_atl.load_image (fn.c_str());
        
        // patient compared with itself
        if (!patient_id.compare(atlas_id))
        {
	        similarity_value_vector[i]=-1;
       	}

        // patient compared with atlas
        else
        {
   
   	    lprintf("Similarity values %s - %s \n", patient_id.c_str(), atlas_id.c_str());
	    similarity_value_vector[i] = compute_nmi_general_score(subject_rtds->get_image().get(), rtds_atl.get_image().get(),
                    parms->atlas_selection_criteria, parms->mi_histogram_bins, mask, parms->lower_mi_value_defined,
                    parms->lower_mi_value, parms->upper_mi_value_defined, parms->upper_mi_value);

	    }
    } 
    
    // Find maximum similarity value
    double max_similarity_value = similarity_value_vector[0];
    for (int i=1; i < number_of_atlases; i++)
    {
	    if (similarity_value_vector[i] > max_similarity_value)
        {
	        max_similarity_value = similarity_value_vector[i];
	    }
    }
    
    // Find min similarity value
    double min_similarity_value = max_similarity_value;
    for (int i=0; i < number_of_atlases; i++)
    {
        if (similarity_value_vector[i] < min_similarity_value && similarity_value_vector[i] != -1)
        {
            min_similarity_value = similarity_value_vector[i];
        }
    }
   
    printf("Minimum similarity value = %g \n", min_similarity_value);
    printf("Maximum similarity value = %g \n", max_similarity_value);
	
    // Find atlas to include in the registration
    std::list<std::string> atlas_most_similar;
    std::list<std::string>::iterator reg_it;
    printf("List of the selected atlases: \n");
    for (reg_it = this->atlas_dir_list.begin(), i=0; 
        reg_it != this->atlas_dir_list.end(); reg_it++, i++)
    {
         std::string atlas = basename (*reg_it);
         if (similarity_value_vector[i] >=
             (((max_similarity_value-min_similarity_value) * parms->mi_percent_thershold) + min_similarity_value))
    	 {
	     atlas_most_similar.push_front(*reg_it);
	     printf("Atlas %s having similarity value = %f \n", atlas.c_str(), similarity_value_vector[i]);
	     }
    }
    printf("Number of selected atlases = %d \n", (int) atlas_most_similar.size());
    
    delete similarity_value_vector;
    
    return atlas_most_similar;
}


std::list<std::string>
Mabs_atlas_selection::random_ranking(std::string patient_id) // Just for testing purpose
{
    lprintf("RANDOM RANKING \n");
    
    std::list<std::string> random_atlases;
    int min_atlases = 4;
    int max_atlases = 12;
    int random_number_of_atlases = rand() % (max_atlases-min_atlases) + min_atlases;
    printf("Selectd %d random atlases \n", random_number_of_atlases);

    int i=0;
    while ((int) random_atlases.size() < random_number_of_atlases) {
        int random_index = rand() % (this->number_of_atlases-min_atlases) + min_atlases;
        std::list<std::string>::iterator atlases_iterator = this->atlas_dir_list.begin();
        std::advance (atlases_iterator, random_index);
        
        if (find(random_atlases.begin(), random_atlases.end(), *atlases_iterator) == random_atlases.end()) {
            i++;
            std::string atlas = basename(*atlases_iterator);
            printf("Atlas number %d is %s \n", i, atlas.c_str());
            random_atlases.push_front(*atlases_iterator);
        }
    }

    return random_atlases;
}


double
Mabs_atlas_selection::compute_nmi_general_score(Plm_image* subject, Plm_image* atlas, std::string score_type,
                                    int hist_bins, MaskTypePointer mask, bool min_value_defined, int min_value,
                                    bool max_value_defined, int max_value)
{
    double score = 0;

    // metric: NMI
    if (score_type == "nmi") 
        {
	    score = compute_nmi(subject, atlas, hist_bins, mask, min_value_defined, min_value,
            max_value_defined, max_value);
	    lprintf("nmi value = %g \n", score);
        }

    // metric: NMI RATIO
    else if (score_type == "nmi-ratio")
        {
	     score = compute_nmi_ratio(subject, atlas, hist_bins, mask, min_value_defined, min_value,
            max_value_defined, max_value);
        }

    return score;
}


double
Mabs_atlas_selection::compute_nmi(Plm_image* img1, Plm_image* img2, int hist_bins, MaskTypePointer mask,
                                    bool min_value_defined, int min_value, bool max_value_defined, int max_value) {
	
    /* Cost function */
    typedef float PixelComponentType;
    const unsigned int Dimension = 3;
    typedef itk::Image< PixelComponentType, Dimension > ImageType;
    typedef itk::NormalizedMutualInformationHistogramImageToImageMetric<ImageType, ImageType > MetricType;
    MetricType::Pointer metric = MetricType::New();
    
    /* Identity transformation */
    typedef itk::TranslationTransform< double, Dimension > TranslationTransformType;
    typedef TranslationTransformType::Pointer TranslationTransformPointer;
    TranslationTransformPointer transform = TranslationTransformType::New();
    transform->SetIdentity();
	
    /* Linear Interpolator */
    typedef itk::LinearInterpolateImageFunction< ImageType, double > InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    
    /* Set mask if defined */
    if (mask) {
       	metric->SetFixedImageMask(mask);	
    }
    
    /* Set histogram interval if defined */
    if (min_value_defined) {
        MetricType::MeasurementVectorType lower_bounds;
        #ifdef ITK4 
        lower_bounds.SetSize(2);
        #endif
        lower_bounds.Fill(min_value);
        metric->SetLowerBound(lower_bounds);
    }

    if (max_value_defined) {
        MetricType::MeasurementVectorType upper_bounds;
        #ifdef ITK4 
        upper_bounds.SetSize(2);
        #endif
        upper_bounds.Fill(max_value);
        metric->SetUpperBound(upper_bounds);
    }
    
    /* Metric settings */
    unsigned int numberOfHistogramBins = hist_bins;
    MetricType::HistogramType::SizeType histogramSize;
    #ifdef ITK4
    histogramSize.SetSize(2);
    #endif
    histogramSize.Fill(numberOfHistogramBins);
    metric->SetHistogramSize(histogramSize);
    
    metric->SetFixedImage(img1->itk_float());
    metric->SetMovingImage(img2->itk_float());
    metric->SetFixedImageRegion(img1->itk_float()->GetLargestPossibleRegion());
    
    metric->SetTransform(transform);
    metric->SetInterpolator(interpolator);
    
    metric->Initialize();
    
    /* MNI compute */
    return (double) metric->GetValue(transform->GetParameters());
}


double
Mabs_atlas_selection::compute_nmi_ratio(Plm_image* subject, Plm_image* atlas, int hist_bins, MaskTypePointer mask,
                                    bool min_value_defined, int min_value, bool max_value_defined, int max_value)
{
    double nmi_pre = compute_nmi(subject, atlas, hist_bins, mask, min_value_defined, min_value,
        max_value_defined, max_value);
    
    lprintf("Similarity value pre = %g \n", nmi_pre);
    
    Registration_parms* regp = new Registration_parms;
    Registration_data* regd = new Registration_data;

    regp->append_stage();
    regp->set_key_val("xform", "bspline", 1);
    regp->set_key_val("optim", "lbfgsb", 1);
    regp->set_key_val("impl", "plastimatch", 1);
    regp->set_key_val("metric", "mi", 1);
    regp->set_key_val("max_its", "10", 1);
    regp->set_key_val("convergence_tol", "0.005", 1);
    regp->set_key_val("grad_tol", "0.001", 1);
    regp->set_key_val("res", "2 2 2", 1);
    regp->set_key_val("grid_spac", "15 15 15", 1);
    
    regd->fixed_image=subject;
    regd->moving_image=atlas;
    
    Plm_image* deformed_atlas = new Plm_image();
    Xform* transformation = new Xform();
    Plm_image_header fixed_pih (regd->fixed_image);
    
    do_registration_pure (&transformation, regd, regp);
    plm_warp (deformed_atlas, 0, transformation, &fixed_pih, regd->moving_image,
        regp->default_value, 0, 1);
    
    double nmi_post = compute_nmi(subject, deformed_atlas, hist_bins, mask, min_value_defined, min_value,
        max_value_defined, max_value);
    
	lprintf("Similarity value post = %g \n", nmi_post);

    return ((nmi_post/nmi_pre)-1) * nmi_post;
}
