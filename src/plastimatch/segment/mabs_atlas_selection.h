/* -----------------------------------------------------------------------
 *    See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
 *       ----------------------------------------------------------------------- */
#ifndef _mabs_atlas_selection_h_
#define _mabs_atlas_selection_h_

#include "plmsegment_config.h"

#include <list>
#include <stdio.h>

#include "itkImageMaskSpatialObject.h"

#include "mabs_parms.h"
#include "plm_image.h"
#include "rt_study.h"


typedef itk::ImageMaskSpatialObject<3> MaskType;
typedef MaskType::Pointer MaskTypePointer;

class PLMSEGMENT_API Mabs_atlas_selection {

public:
    Mabs_atlas_selection(std::list<std::string> atlases_list, Rt_study::Pointer subject);

    ~Mabs_atlas_selection();
    
    std::list<std::string> nmi_ranking(std::string patient_id, const Mabs_parms* parms);

    std::list<std::string> random_ranking(std::string patient_id); // Just for testing purpose

    double compute_nmi(Plm_image* img1, Plm_image* img2, int hist_bins, MaskTypePointer mask,
                       bool min_value_defined, int min_value, bool max_value_defined, int max_value);

    double compute_nmi_ratio(Plm_image* subject, Plm_image* atlas, int hist_bins, MaskTypePointer mask,
                       bool min_value_defined, int min_value, bool max_value_defined, int max_value);

    double compute_nmi_general_score(Plm_image* subject, Plm_image* atlas, std::string score_type, int hist_bins,
                       MaskTypePointer mask, bool min_value_defined, int min_value, bool max_value_defined,
                       int max_value);

public:
    Rt_study::Pointer subject_rtds;
    std::list<std::string> atlas_dir_list;
    int number_of_atlases; 
};

#endif /* #ifndef _mabs_atlases_selection_h_ */
