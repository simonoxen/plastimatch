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
#include "string_util.h"

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

    lprintf("MI RANKING \n");
	
    printf ("NUMBER OF INITIAL ATLASES = %d \n", number_of_atlases);
    double* nmi_vector = new double[number_of_atlases];
    
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
        
        if (!patient_id.compare(atlas_id))
        {
	    nmi_vector[i]=-1;
       	}
        else
	{
	    nmi_vector[i] = compute_nmi(subject_rtds->get_image().get(), rtds_atl.get_image().get(),
                            parms->mi_histogram_bins, mask, parms->lower_mi_value_defined, parms->lower_mi_value,
                            parms->upper_mi_value_defined, parms->upper_mi_value);
	    lprintf("NMI %s - %s = %g \n", patient_id.c_str(), atlas_id.c_str(), nmi_vector[i]);
	}
    } 
    
    // Find maximum nmi
    double max_nmi = nmi_vector[0];
    for (int i=1; i < number_of_atlases; i++)
    {
	if (nmi_vector[i] > max_nmi)
        {
	    max_nmi = nmi_vector[i];
	}
    }
    
    // Find min nmi
    double min_nmi = max_nmi;
    for (int i=0; i < number_of_atlases; i++)
    {
        if (nmi_vector[i] < min_nmi && nmi_vector[i] != -1)
        {
            min_nmi = nmi_vector[i];
        }
    }
   
    printf("MINIMUM NMI = %g \n", min_nmi);
    printf("MAXIMUM NMI = %g \n", max_nmi);
	
    // Find atlas to include in the registration
    std::list<std::string> atlas_most_similar;
    std::list<std::string>::iterator reg_it;
    printf("LIST OF THE SELECTED ATLASES: \n");
    for (reg_it = this->atlas_dir_list.begin(), i=0; 
        reg_it != this->atlas_dir_list.end(); reg_it++, i++)
    {
         std::string atlas = basename (*reg_it);
         if (nmi_vector[i] >= (((max_nmi-min_nmi) * parms->mi_percent_thershold) + min_nmi))
	 {
	     atlas_most_similar.push_front(*reg_it);
	     printf("ATLAS %s HAVING NMI = %f \n", atlas.c_str(), nmi_vector[i]);
	 }
    }
    printf("NUMBER OF SELECTED ATLASES = %d \n", (int) atlas_most_similar.size());
    
    delete nmi_vector;
    
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
