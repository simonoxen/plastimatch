/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_atlas_selection_h_
#define _mabs_atlas_selection_h_

#include "plmsegment_config.h"
#include <list>
#include <stdio.h>
#include "itkImageMaskSpatialObject.h"

#include "mabs_parms.h"
#include "plm_image.h"

typedef itk::ImageMaskSpatialObject<3> MaskType;
typedef MaskType::Pointer MaskTypePointer;

class PLMSEGMENT_API Mabs_atlas_selection {

public:
    Mabs_atlas_selection();
    ~Mabs_atlas_selection();
    void run_selection();
    void nmi_ranking();
    double compute_nmi_general_score();
    double compute_nmi_ratio();
    double compute_nmi (
        const Plm_image::Pointer& img1, 
        const Plm_image::Pointer& img2);
    void random_ranking(); /* Just for testing purpose */
    void precomputed_ranking(); /* Just for testing purpose */

public:
    Plm_image::Pointer subject;
    std::string subject_id;
    std::list<std::string> atlas_dir_list;
    int number_of_atlases;
    Plm_image::Pointer atlas;
    const Mabs_parms* atlas_selection_parms;
    int hist_bins;
    MaskTypePointer mask;
    bool min_value_defined;
    int min_value;
    bool max_value_defined;
    int max_value;
    std::list<std::pair<std::string, double> > selected_atlases;
};

#endif /* #ifndef _mabs_atlases_selection_h_ */
