/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegion.h"

//#include "gamma_analysis.h"
#include "gamma_dose_comparison.h"
#include "itk_resample.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"

class Gamma_parms;

void do_gamma_threshold (Gamma_parms *parms);


/*! \enum Gamma_output_mode Selector for output image type (gamma, or binary pass/fail)
*/
enum Gamma_labelmap_mode {
    NONE,  /*!< disable output binary uchar labelmap for gamma */
    PASS,  /*!< output binary (1/0) image of type uchar, 1 if gamma<1 */ 
    FAIL   /*!< output binary (1/0) image of type uchar, 1 if gamma>1 */ 
};

/*! \class Gamma_parms
    \brief This is the Gamma_parms class.
    * Used to pass input and output parameters for gamma analysis
	to/from find_dose_threshold() and do_gamma_analysis() */
class Gamma_parms {
public:
    
    float r_tol, /*!< distance-to-agreement (DTA) criterion, input parameter */ 
        d_tol, /*!< dose-difference criterion, input for do_gamma_analysis, in Gy 
                 Set from 3D Slicer plugin either directly or as percentrage of
                 dose_max, as found by find_dose_threshold().
               */
    //        dose_max, /*!< maximum dose (max voxel value) in the img_in1, set by find_dose_threshold() */
        gamma_max; /*!< maximum gamma to calculate */

    Plm_image *img_in1; /*!< input dose image 1 for gamma analysis*/
    Plm_image *img_in2; /*!< input dose image 2 for gamma analysis*/
    Plm_image *img_out; /*!< output float type image, voxel value = calculated gamma value */
    Plm_image *labelmap_out; /*!< output uchar type labelmap, voxel value = 1/0 for pass/fail */

    Gamma_labelmap_mode mode; /*!< output mode selector for 3D Slicer plugin*/

public:
    Gamma_parms () { /*!< Constructor for Gamma_parms, sets default values (mode GAMMA) 
                       for the 3D Slicer plugin */
        img_in1 = 0;
        img_in2 = 0;
        img_out = 0;
        labelmap_out = 0;
        r_tol = d_tol = gamma_max = 3; 
        mode = NONE;
    }
};


class Gamma_dose_comparison_private {
public:
    Gamma_dose_comparison_private () {
        have_gamma_image = false;

        have_reference_dose = false;
        reference_dose = 0.f;

        dose_max = 0.f;

        have_analysis_thresh = false;
        analysis_thresh = 0.f;
        analysis_num_vox = 0;
        analysis_num_pass = 0;
    }
public:
    Gamma_parms gp;
    bool have_gamma_image;

    /* reference dose value, used for gamma analysis and analysis 
       thresholding.  */
    bool have_reference_dose;
    float reference_dose;

    /* maximum dose (max voxel value) in the reference dose, 
      set by find_reference_max_dose() */
    float dose_max;

    /* analysis thresholding, limits statistics to dose values above 
       the threshold */
    bool have_analysis_thresh;
    float analysis_thresh;
    plm_long analysis_num_vox;
    plm_long analysis_num_pass;
public:
    void find_reference_max_dose ();
    void do_gamma_analysis ();
};

Gamma_dose_comparison::Gamma_dose_comparison () {
    d_ptr = new Gamma_dose_comparison_private;
}

Gamma_dose_comparison::~Gamma_dose_comparison () {
    delete d_ptr;
}

void 
Gamma_dose_comparison::set_reference_image (const char* image_fn)
{
    d_ptr->gp.img_in1 = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_reference_image (Plm_image* image)
{
    d_ptr->gp.img_in1 = image;
}

void 
Gamma_dose_comparison::set_reference_image (
    const FloatImageType::Pointer image)
{
    d_ptr->gp.img_in1 = new Plm_image (image);
}

void 
Gamma_dose_comparison::set_compare_image (const char* image_fn)
{
    d_ptr->gp.img_in2 = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_compare_image (Plm_image* image)
{
    d_ptr->gp.img_in2 = image;
}

void 
Gamma_dose_comparison::set_compare_image (
    const FloatImageType::Pointer image)
{
    d_ptr->gp.img_in2 = new Plm_image (image);
}

float
Gamma_dose_comparison::get_spatial_tolerance ()
{
    return d_ptr->gp.r_tol;
}

void 
Gamma_dose_comparison::set_spatial_tolerance (float spatial_tol)
{
    d_ptr->gp.r_tol = spatial_tol;
}

float
Gamma_dose_comparison::get_dose_difference_tolerance ()
{
    return d_ptr->gp.d_tol;
}

void 
Gamma_dose_comparison::set_dose_difference_tolerance (float dose_tol)
{
    d_ptr->gp.d_tol = dose_tol;
}

void 
Gamma_dose_comparison::set_reference_dose (float dose)
{
    d_ptr->reference_dose = dose;
    d_ptr->have_reference_dose = true;
}

void 
Gamma_dose_comparison::set_analysis_threshold (float thresh)
{
    d_ptr->have_analysis_thresh = true;
    d_ptr->analysis_thresh = thresh;
}

void 
Gamma_dose_comparison::set_gamma_max (float gamma_max)
{
    d_ptr->gp.gamma_max = gamma_max;
}

void 
Gamma_dose_comparison::run ()
{
    if (!d_ptr->have_reference_dose) {
        d_ptr->find_reference_max_dose ();
        d_ptr->reference_dose = d_ptr->dose_max;
    }
    d_ptr->have_gamma_image = true;
    resample_image_to_reference (d_ptr->gp.img_in1, d_ptr->gp.img_in2);
    d_ptr->do_gamma_analysis ();
}

Plm_image*
Gamma_dose_comparison::get_gamma_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    return d_ptr->gp.img_out;
}

FloatImageType::Pointer
Gamma_dose_comparison::get_gamma_image_itk ()
{
    return get_gamma_image()->itk_float();
}

Plm_image*
Gamma_dose_comparison::get_pass_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    d_ptr->gp.mode = PASS;
    do_gamma_threshold (&d_ptr->gp);
    return d_ptr->gp.labelmap_out;
}

UCharImageType::Pointer
Gamma_dose_comparison::get_pass_image_itk ()
{
    return get_pass_image()->itk_uchar();
}

Plm_image*
Gamma_dose_comparison::get_fail_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    d_ptr->gp.mode = FAIL;
    do_gamma_threshold (&d_ptr->gp);
    return d_ptr->gp.labelmap_out;
}

UCharImageType::Pointer
Gamma_dose_comparison::get_fail_image_itk ()
{
    return get_fail_image()->itk_uchar();
}

float
Gamma_dose_comparison::get_pass_fraction ()
{
    if (d_ptr->analysis_num_vox < 1) {
        return 0.f;
    }
    return d_ptr->analysis_num_pass / (float) d_ptr->analysis_num_vox;
}

void 
Gamma_dose_comparison::resample_image_to_reference (
    Plm_image *image_reference, Plm_image *image_moving)
{
    Plm_image_header pih;
    pih.set_from_plm_image (image_reference);
    itk::Image<float, 3>::Pointer resampledMovingImage = resample_image (
        image_moving->itk_float(),
        &pih,
        0.f,
        false
    );

    image_moving->set_itk(resampledMovingImage);
}

/* -------------------------------------------------------------------------
   from gamma_analysis.cxx
   ------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
   Private functions
   ------------------------------------------------------------------------- */
void 
Gamma_dose_comparison_private::find_reference_max_dose ()
{
    FloatImageType::Pointer img_in1 = gp.img_in1->itk_float();
    typedef itk::ImageRegionIteratorWithIndex< 
        FloatImageType > FloatIteratorType;
    typedef itk::ImageRegion<3> FloatRegionType;
    
    FloatRegionType all_of_img1 = img_in1->GetLargestPossibleRegion();
    FloatIteratorType img_in1_iterator (img_in1, all_of_img1);
    float maxlevel1=-1e20;
    for (img_in1_iterator.GoToBegin(); 
         !img_in1_iterator.IsAtEnd(); 
         ++img_in1_iterator)
    {
        float level1 = img_in1_iterator.Get();
        if (level1 > maxlevel1) maxlevel1 = level1;         
    } 
    this->dose_max = maxlevel1;
}

void 
Gamma_dose_comparison_private::do_gamma_analysis ()
{ 
    float spacing_in[3], origin_in[3];
    plm_long dim_in[3];
    Plm_image_header pih;
    float gamma;

    FloatImageType::Pointer img_in1 = gp.img_in1->itk_float();
    FloatImageType::Pointer img_in2 = gp.img_in2->itk_float();

    pih.set_from_itk_image (img_in1);
    pih.get_dim (dim_in );
    pih.get_origin (origin_in );
    pih.get_spacing (spacing_in );

    // Create ITK image for gamma output, "pass", "fail" and combined 
    FloatImageType::SizeType sz;
    FloatImageType::IndexType st;
    FloatImageType::RegionType rg;
    FloatImageType::PointType og;
    FloatImageType::SpacingType sp;
    FloatImageType::DirectionType dc;
    for (int d1 = 0; d1 < 3; d1++) {
        st[d1] = 0;
        sz[d1] = dim_in[d1];
        sp[d1] = spacing_in[d1];
        og[d1] = origin_in[d1];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);
    dc = pih.m_direction;

    FloatImageType::Pointer gamma_img = FloatImageType::New();
    UCharImageType::Pointer gamma_labelmap = UCharImageType::New();

    gamma_img->SetRegions (rg);
    gamma_img->SetOrigin (og);
    gamma_img->SetSpacing (sp);
    gamma_img->SetDirection (dc);
    gamma_img->Allocate();

    gamma_labelmap->SetRegions (rg);
    gamma_labelmap->SetOrigin (og);
    gamma_labelmap->SetSpacing (sp);
    gamma_labelmap->SetDirection (dc);
    gamma_labelmap->Allocate();

    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > UCharIteratorType;
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > FloatIteratorType;
    typedef itk::ImageRegion<3> FloatRegionType;
    
    FloatRegionType all_of_img1 = img_in1->GetLargestPossibleRegion();
    FloatRegionType all_of_img2 = img_in2->GetLargestPossibleRegion();
    FloatRegionType subset_of_img2;

    FloatIteratorType img_in1_iterator (img_in1, all_of_img1);
    FloatIteratorType gamma_img_iterator (gamma_img, gamma_img->GetLargestPossibleRegion());
    UCharIteratorType gamma_labelmap_iterator (gamma_labelmap, gamma_labelmap->GetLargestPossibleRegion());

    FloatImageType::IndexType k1, k2, k3;
    FloatImageType::OffsetType offset;
    FloatImageType::SizeType region_size;
    FloatPoint3DType phys;

    int reg_pixsize; 
    float level1, level2, dr2, dd2, gg;
    float f0,f1,f2,f3;

    //vox-to-mm-to-gamma conversion factors; strictly, these should come from IMAGE2, not 1
    f0 = spacing_in[0]/gp.r_tol; f0=f0*f0;
    f1 = spacing_in[1]/gp.r_tol; f1=f1*f1;
    f2 = spacing_in[2]/gp.r_tol; f2=f2*f2;
    f3 = 1./gp.d_tol; f3 = f3*f3;

    // get min spacing, safeguard against negative spacings.
    float min_spc = fabs(spacing_in[0]);
    if (fabs(spacing_in[1])<min_spc) min_spc=fabs(spacing_in[1]);
    if (fabs(spacing_in[2])<min_spc) min_spc=fabs(spacing_in[2]);

    // if gamma is limited to gamma_max, 
    // no need to look beyond reg_pixsize away from the current voxel
    reg_pixsize = (int)ceil(1+  gp.gamma_max/min_spc);
    
    offset[0]=reg_pixsize;
    offset[1]=reg_pixsize;
    offset[2]=reg_pixsize;

    float analysis_threshold = this->analysis_thresh * this->reference_dose;

    gamma_img_iterator.GoToBegin();
    gamma_labelmap_iterator.GoToBegin();

    for (img_in1_iterator.GoToBegin(); 
         !img_in1_iterator.IsAtEnd(); 
         ++img_in1_iterator)
    {
        //calculate gamma for this voxel of input image
        level1 = img_in1_iterator.Get();
        k1=img_in1_iterator.GetIndex();
        img_in1->TransformIndexToPhysicalPoint( k1, phys );
        img_in2->TransformPhysicalPointToIndex( phys, k2 );

        //k2 is the voxel index of the k1's physical (mm) position in img2
    
        // set subset_of_img2 to the following region:  
        // k1[0]-region_size < k2[0] < k1[0]+region_size, same for y,z
        // assume (approx) same pix spacing in img1 and img2
        // crop the region by the entire image to be safe
        k2 -= offset;  
        subset_of_img2.SetIndex(k2);
        region_size.Fill( 2*reg_pixsize ); 
        subset_of_img2.SetSize( region_size);
        subset_of_img2.Crop( all_of_img2 );

        FloatIteratorType img_in2_iterator (img_in2, subset_of_img2);

        // calculate gamma, take a minimum of ... over the subset_of_img2
        gamma = 1e20;
        for (img_in2_iterator.GoToBegin(); 
             !img_in2_iterator.IsAtEnd(); 
             ++img_in2_iterator)
        {
            k3 = img_in2_iterator.GetIndex();
            level2 = img_in2_iterator.Get();
            dr2 = (k3[0]-k1[0])*(k3[0]-k1[0])*f0 +
                (k3[1]-k1[1])*(k3[1]-k1[1])*f1 +
                (k3[2]-k1[2])*(k3[2]-k1[2])*f2 ;
            dd2 = (level1 - level2)*(level1-level2)*f3;
            gg = dr2 + dd2;
            if (gg < gamma) gamma=gg;
            //test only: if (k1[0]==k3[0]) gamma = k3[0];
        }
        gamma = sqrt(gamma);
        if (gamma > gp.gamma_max) {
            gamma = gp.gamma_max;
        }
        gamma_img_iterator.Set (gamma);

        /* Get statistics */
        if (this->have_analysis_thresh) {
            if (level1 > analysis_threshold) {
                this->analysis_num_vox ++;
                if (gamma <= 1) {
                    this->analysis_num_pass ++;
                }
            }
        }
    }

    gp.img_out = new Plm_image;
    gp.img_out->set_itk (gamma_img);
}

static
void 
do_gamma_threshold (Gamma_parms *parms)
{ 
    FloatImageType::Pointer ref_img = parms->img_in1->itk_float();
    FloatImageType::Pointer gamma_img = parms->img_out->itk_float();

    /* Create labelmap image if not already created */
    if (!parms->labelmap_out) {
        parms->labelmap_out = new Plm_image;
        UCharImageType::Pointer gamma_labelmap = UCharImageType::New();
        itk_image_header_copy (gamma_labelmap, gamma_img);
        gamma_labelmap->Allocate();
        parms->labelmap_out = new Plm_image (gamma_labelmap);
    }
    UCharImageType::Pointer gamma_labelmap = parms->labelmap_out->itk_uchar();

    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > 
        UCharIteratorType;
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > 
        FloatIteratorType;
    typedef itk::ImageRegion<3> FloatRegionType;
    
    FloatIteratorType ref_it (gamma_img, 
        ref_img->GetLargestPossibleRegion());
    FloatIteratorType gam_it (gamma_img, 
        gamma_img->GetLargestPossibleRegion());
    UCharIteratorType lab_it (gamma_labelmap,
        gamma_labelmap->GetLargestPossibleRegion());

    /* Loop through gamma image, compare against threshold */
    for (ref_it.GoToBegin(), gam_it.GoToBegin(), lab_it.GoToBegin(); 
         !ref_it.IsAtEnd(); 
         ++ref_it, ++gam_it, ++lab_it)
    {
        float ref_dose = ref_it.Get();
        float gamma = gam_it.Get();
        switch (parms->mode) {
        case PASS:
            if ((gamma >=0) && (gamma <= 1) && ref_dose > 0) {
                lab_it.Set (1);
            } else {
                lab_it.Set (0);
            }
            break;
        case FAIL:
            if (gamma > 1) {
                lab_it.Set (1);
            } else {
                lab_it.Set (0);
            }
            break;
        case NONE:
        default:
            lab_it.Set (0);
            break;
        }
    }
}
