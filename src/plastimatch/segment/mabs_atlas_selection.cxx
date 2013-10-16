/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdlib.h>

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
#include "rt_study.h"
#include "xform.h"

Mabs_atlas_selection::Mabs_atlas_selection ()
{
    /* constructor */
    this->hist_bins = 100;
    this->atlas_selection_parms = NULL;
    this->mask = NULL;
    this->min_hist_sub_value_defined = false;
    this->min_hist_sub_value=0;
    this->max_hist_sub_value_defined = false;
    this->max_hist_sub_value=0;
    this->min_hist_atl_value_defined = false;
    this->min_hist_atl_value=0;
    this->max_hist_atl_value_defined = false;
    this->max_hist_atl_value=0;
}


Mabs_atlas_selection::~Mabs_atlas_selection ()
{
    /* destructor */
}


void
Mabs_atlas_selection::run_selection()
{
    if (this->atlas_selection_parms->atlas_selection_criteria == "nmi" ||
        this->atlas_selection_parms->atlas_selection_criteria == "nmi-ratio")
    {
        this->nmi_ranking();
    }

    else if (this->atlas_selection_parms->atlas_selection_criteria == "random")
    {
        this->random_ranking ();
    }

    else if (this->atlas_selection_parms->atlas_selection_criteria == "precomputed")
    {
        this->precomputed_ranking ();
    }

}

void
Mabs_atlas_selection::nmi_ranking()
{
    lprintf("NMI RANKING \n");

    this->hist_bins = this->atlas_selection_parms->mi_histogram_bins;
    this->min_hist_sub_value_defined = this->atlas_selection_parms->lower_mi_value_sub_defined;
    this->min_hist_sub_value = this->atlas_selection_parms->lower_mi_value_sub;
    this->max_hist_sub_value_defined = this->atlas_selection_parms->upper_mi_value_sub_defined;
    this->max_hist_sub_value = this->atlas_selection_parms->upper_mi_value_sub;
    this->min_hist_atl_value_defined = this->atlas_selection_parms->lower_mi_value_atl_defined;
    this->min_hist_atl_value = this->atlas_selection_parms->lower_mi_value_atl;
    this->max_hist_atl_value_defined = this->atlas_selection_parms->upper_mi_value_atl_defined;
    this->max_hist_atl_value = this->atlas_selection_parms->upper_mi_value_atl;

    printf ("Number of initial atlases = %d \n", this->number_of_atlases);
    double* similarity_value_vector = new double[this->number_of_atlases];
    
    /* Set the mask if defined */
    if (this->atlas_selection_parms->roi_mask_fn.compare("")!=0)
    {
       	Plm_image* mask_plm = plm_image_load (this->atlas_selection_parms->roi_mask_fn, PLM_IMG_TYPE_ITK_FLOAT);
    	
       	this->mask = MaskType::New();
       	this->mask->SetImage(mask_plm->itk_uchar());
       	this->mask->Update();
        
        delete mask_plm;
    }
	
    /* Loop through images in the atlas */
    std::list<std::string>::iterator atl_it;
    int i;
    for (atl_it = this->atlas_dir_list.begin(), i=0;
         atl_it != this->atlas_dir_list.end(); atl_it++, i++)
    {	
        Rt_study* rtds_atl= new Rt_study;
       	std::string path_atl = *atl_it;
        std::string atlas_id = basename (path_atl);
        std::string atlas_input_path = string_format ("%s/prealign/%s", "training-mgh", atlas_id.c_str());
        std::string fn = string_format ("%s/img.nrrd", atlas_input_path.c_str());
        rtds_atl->load_image (fn.c_str());
        
        this->atlas = rtds_atl->get_image();

        /* subject compared with itself */
        if (!this->subject_id.compare(atlas_id))
        {
            similarity_value_vector[i]=-1;
       	}

        /* subject compared with atlas */
        else
        {
            lprintf("Similarity values %s - %s \n", this->subject_id.c_str(), atlas_id.c_str());
            similarity_value_vector[i] = this->compute_nmi_general_score();
        }
       
        delete rtds_atl;
    } 
   
    if (!mask) mask = NULL;
    
    /* Find maximum similarity value */
    double max_similarity_value = similarity_value_vector[0];
    for (int i=1; i < number_of_atlases; i++)
    {
        if (similarity_value_vector[i] > max_similarity_value)
        {
            max_similarity_value = similarity_value_vector[i];
        }
    }
    
    /* Find min similarity value */
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
	
    /* Find atlas to include in the registration */
    std::list<std::pair<std::string, double> > most_similar_atlases;
    std::list<std::string>::iterator reg_it;
    printf("List of the selected atlases: \n");
    for (reg_it = this->atlas_dir_list.begin(), i=0; 
        reg_it != this->atlas_dir_list.end(); reg_it++, i++)
    {
        std::string atlas = basename (*reg_it);
        if (similarity_value_vector[i] >=
            (((max_similarity_value-min_similarity_value) * this->atlas_selection_parms->mi_percent_threshold) + min_similarity_value))
    	{
            most_similar_atlases.push_back(std::make_pair(atlas, (double) similarity_value_vector[i]));
            printf("Atlas %s having similarity value = %f \n", atlas.c_str(), similarity_value_vector[i]);
        }
    }
    printf("Number of selected atlases = %d \n", (int) most_similar_atlases.size());
    
    delete similarity_value_vector;
    
    this->selected_atlases = most_similar_atlases;
}


double
Mabs_atlas_selection::compute_nmi_general_score()
{
    double score = 0;

    /* metric: NMI */
    if (this->atlas_selection_parms->atlas_selection_criteria == "nmi") 
    {
        score = this->compute_nmi(this->subject, this->atlas);
        lprintf("nmi value = %g \n", score);
    }

    /* metric: NMI RATIO */
    else if (this->atlas_selection_parms->atlas_selection_criteria == "nmi-ratio")
    {
        score = this->compute_nmi_ratio();
    }

    return score;
}


double
Mabs_atlas_selection::compute_nmi_ratio()
{
    double nmi_pre = this->compute_nmi(this->subject, this->atlas);
    
    lprintf("Similarity value pre = %g \n", nmi_pre);
    
    Registration_parms* regp = new Registration_parms;
    Registration_data* regd = new Registration_data;

    regp->parse_command_file(this->atlas_selection_parms->nmi_ratio_registration_config_fn.c_str());
    
    regd->fixed_image = this->subject;
    regd->moving_image = this->atlas;

    /* Make sure to have just the right inputs */
    regd->fixed_landmarks = NULL;
    regd->moving_landmarks = NULL;

    Plm_image::Pointer deformed_atlas = Plm_image::New ();
    Xform* transformation = new Xform();
    Plm_image_header fixed_pih (regd->fixed_image);

    do_registration_pure (&transformation, regd, regp);
    plm_warp (deformed_atlas.get(), 0, transformation, &fixed_pih, 
        regd->moving_image.get(),
        regp->default_value, 0, 1);
    
    double nmi_post = this->compute_nmi (this->subject, deformed_atlas);
    
    delete regd;
    delete regp;
    delete transformation;

    lprintf("Similarity value post = %g \n", nmi_post);

    return ((nmi_post/nmi_pre)-1) * nmi_post;
}


double
Mabs_atlas_selection::compute_nmi (
    const Plm_image::Pointer& img1, /* Subject */ 
    const Plm_image::Pointer& img2) /* Atlas */
{	
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
    if (this->mask)
    {
       	metric->SetFixedImageMask(this->mask);	
    }
    
    /* Set histogram interval if defined */
    if (this->min_hist_sub_value_defined &&
        this->min_hist_atl_value_defined)
    {
        MetricType::MeasurementVectorType lower_bounds;
        #ifdef ITK4 
        lower_bounds.SetSize(2);
        #endif
        lower_bounds.SetElement(0, this->min_hist_sub_value);
        lower_bounds.SetElement(1, this->min_hist_atl_value);
        metric->SetLowerBound(lower_bounds);
    }

    if (this->max_hist_sub_value_defined &&
        this->max_hist_atl_value_defined)
    {
        MetricType::MeasurementVectorType upper_bounds;
        #ifdef ITK4 
        upper_bounds.SetSize(2);
        #endif
        upper_bounds.SetElement(0, this->max_hist_sub_value);
        upper_bounds.SetElement(1, this->max_hist_atl_value);
        metric->SetUpperBound(upper_bounds);
    }
    
    /* Metric settings */
    unsigned int numberOfHistogramBins = this->hist_bins;
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


void
Mabs_atlas_selection::random_ranking() /* Just for testing purpose */
{
    lprintf("RANDOM RANKING \n");
    
    std::list<std::string> random_atlases;

    int min_atlases = this->atlas_selection_parms->min_random_atlases;
    int max_atlases = this->atlas_selection_parms->max_random_atlases + 1;

    int random_number_of_atlases = rand() % (max_atlases-min_atlases) + min_atlases;
    printf("Selected %d random atlases for the subject %s \n", random_number_of_atlases, this->subject_id.c_str());

    /* Select the random atlases */
    int i=0;
    while ((int) random_atlases.size() < random_number_of_atlases) {

        /* Choose a random atlas */
        int random_index = rand() % (this->number_of_atlases);
        std::list<std::string>::iterator atlases_iterator = this->atlas_dir_list.begin();
        std::advance (atlases_iterator, random_index);

        /* Check if the random atlas has been already included and if it is different from the subject */
        if (find(random_atlases.begin(), random_atlases.end(), basename(*atlases_iterator)) == random_atlases.end() &&
            basename(*atlases_iterator) != this->subject_id)
        {
            i++;
            std::string atlas = basename(*atlases_iterator);
            printf("Atlas number %d is %s \n", i, atlas.c_str());
            random_atlases.push_back(atlas);
        } 
    }
    
    /* Fill this->selected_atlas */
    std::list<std::string>::iterator atl_it;
    for (atl_it = random_atlases.begin(); 
        atl_it != random_atlases.end(); atl_it++)
    {
        this->selected_atlases.push_back(std::make_pair(*atl_it, 0.0));
    }
 
}


void
Mabs_atlas_selection::precomputed_ranking() /* Just for testing purpose */
{
    /* Open the file where is stored the precomputed ranking*/
    std::ifstream ranking_file (this->atlas_selection_parms->precomputed_ranking_fn.c_str());
    
    std::string atlases_line;

    /* Read line by line*/
    while(std::getline(ranking_file, atlases_line))
    {
      
        std::istringstream atlases_line_stream (atlases_line);
        std::string item;
        
        /* item_index is equal to 0 for the first element in the line (the subject id) and
         * greater than 0 for the atlases */
        int item_index = -1;

        /* Split a line using the spaces */
        while(std::getline(atlases_line_stream, item, ' '))
        {

            /* If defined select the first n atlases from the ranking */
            if (this->atlas_selection_parms->atlases_from_precomputed_ranking_defined &&
                (int) this->selected_atlases.size() >=
                this->atlas_selection_parms->atlases_from_precomputed_ranking)
            {
                break;
            }
            
            item_index++;
            
            /* Delete space, colon and equals sign from single item name */
            char chars_to_remove [] = " :=";
            for (unsigned int i = 0; i < strlen(chars_to_remove); i++)
            {
                item.erase(std::remove(item.begin(), item.end(), chars_to_remove[i]), item.end());
            }

            /* First element in the line is the subject */
            if (item_index == 0) {

                /* This row doesn't contain data regarding the current subject, jump it */
                if (item != this->subject_id) break; 
                 
                /* This row contains the data regarding the current subject,
                 * jump just the subject id */
                else if (item == this->subject_id) continue;
            }
            
            /* After the first element there are the selected atlases */
            else
            {
                this->selected_atlases.push_back(std::make_pair(item, 0.0));
            }
        }
    }
    
    /* Close log file */    
    ranking_file.close();
}
