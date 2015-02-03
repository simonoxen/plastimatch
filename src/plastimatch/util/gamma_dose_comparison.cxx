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

#include "gamma_dose_comparison.h"
#include "itk_resample.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"
//#include "geometry_chooser.h"


/*! \enum Gamma_output_mode Selector for output image type (gamma, or binary pass/fail)
*/
enum Gamma_labelmap_mode {
    NONE,  /*!< disable output binary uchar labelmap for gamma */
    PASS,  /*!< output binary (1/0) image of type uchar, 1 if gamma<1 */ 
    FAIL   /*!< output binary (1/0) image of type uchar, 1 if gamma>1 */ 
};

class Gamma_dose_comparison_private {
public:
    Gamma_dose_comparison_private ()
    {
        img_in1 = 0;
        img_in2 = 0;
        img_mask = 0;
        labelmap_out = 0;

        have_gamma_image = false;
        gamma_image = Plm_image::New();

        dta_tolerance = 3.0;
        dose_difference_tolerance = 0.03;
        gamma_max = 2.0;
        mode = NONE;

        have_reference_dose = false;
        reference_dose = 0.f;

        dose_max = 0.f;

        have_analysis_thresh = false;
        analysis_thresh = 0.f;
        analysis_num_vox = 0;
        analysis_num_pass = 0;
		
		str_gamma_report ="";
		b_local_gamma = false;
		b_skip_low_dose_gamma = true;
		f_inherent_resample_mm = -1.0;
		i_total_vox_num = 0;
    }
public:
    Plm_image *img_in1; /*!< input dose image 1 for gamma analysis*/
    Plm_image *img_in2; /*!< input dose image 2 for gamma analysis*/
    Plm_image *img_mask; /*!< input mask image for gamma analysis*/
    Plm_image *labelmap_out; /*!< output uchar type labelmap, voxel value = 1/0 for pass/fail */

    /* Gamma image is float type image, voxel value = calculated gamma value */
    bool have_gamma_image;
    Plm_image::Pointer gamma_image;

    /* distance-to-agreement (DTA) criterion, input parameter */ 
    float dta_tolerance;
    /* dose-difference criterion, in percent.  Either as percent of 
       prescription (as requested by user), or as percent of 
       computed dose_max */
    float dose_difference_tolerance;
    /* maximum gamma to calculate */
    float gamma_max;
    Gamma_labelmap_mode mode; /*!< output mode selector for 3D Slicer plugin*/

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

	/*extended features by YK*/
	std::string str_gamma_report;
	bool b_local_gamma;
	bool b_skip_low_dose_gamma;
	float f_inherent_resample_mm;
	int i_total_vox_num; //dim[0]*dim[1]*dim[2] //this is just for report


public:
    void do_mask_threshold ();
    void find_reference_max_dose ();
    void do_gamma_analysis ();
    void do_gamma_threshold ();
	void compose_report();//fill str_gamma_report;
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
    d_ptr->img_in1 = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_reference_image (Plm_image* image)
{
    d_ptr->img_in1 = image;
}

void 
Gamma_dose_comparison::set_reference_image (
    const FloatImageType::Pointer image)
{
    d_ptr->img_in1 = new Plm_image (image);
}

void 
Gamma_dose_comparison::set_compare_image (const char* image_fn)
{
    d_ptr->img_in2 = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_compare_image (Plm_image* image)
{
    d_ptr->img_in2 = image;
}

void 
Gamma_dose_comparison::set_compare_image (
    const FloatImageType::Pointer image)
{
    d_ptr->img_in2 = new Plm_image (image);
}

void 
Gamma_dose_comparison::set_mask_image (const char* image_fn)
{
  d_ptr->img_mask = new Plm_image (image_fn);
}

void 
Gamma_dose_comparison::set_mask_image (Plm_image* image)
{
  d_ptr->img_mask = image;
}

void 
Gamma_dose_comparison::set_mask_image (
  const UCharImageType::Pointer image)
{
  d_ptr->img_mask = new Plm_image (image);
}

float
Gamma_dose_comparison::get_spatial_tolerance ()
{
    return d_ptr->dta_tolerance;
}

void 
Gamma_dose_comparison::set_spatial_tolerance (float spatial_tol)
{
    d_ptr->dta_tolerance = spatial_tol;
}

float
Gamma_dose_comparison::get_dose_difference_tolerance ()
{
    return d_ptr->dose_difference_tolerance;
}

void 
Gamma_dose_comparison::set_dose_difference_tolerance (float dose_tol)
{
    d_ptr->dose_difference_tolerance = dose_tol;
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
    d_ptr->analysis_thresh = thresh; //0.1 = 10%
}

void 
Gamma_dose_comparison::set_gamma_max (float gamma_max)
{
    d_ptr->gamma_max = gamma_max;
}

void 
Gamma_dose_comparison::run ()
{
    if (!d_ptr->have_reference_dose) {
        d_ptr->find_reference_max_dose ();
        d_ptr->reference_dose = d_ptr->dose_max;
    }
    d_ptr->have_gamma_image = true;

	//Edited by YK
	//if the reference image is too sparse, resample it with smaller spacing. (e.g. 1 mm)
	if (d_ptr->f_inherent_resample_mm > 0.0){		
		float spacing[3];
		spacing[0] = d_ptr->f_inherent_resample_mm;
		spacing[1] = d_ptr->f_inherent_resample_mm;
		spacing[2] = d_ptr->f_inherent_resample_mm;
		resample_image_with_fixed_spacing(d_ptr->img_in1, spacing);		

		//std::cout << "Self-resampling is done" <<std::endl;
	}	

    // Threshold mask image to have values 1 and 0 and resample it to reference
    if (d_ptr->img_mask) {
        d_ptr->do_mask_threshold ();
        resample_image_to_reference (d_ptr->img_in1, d_ptr->img_mask);		
    }	

	//YK added
	//if resample value_mm > 0, 
	//first resample the reference image using that value then go ahead	
    resample_image_to_reference (d_ptr->img_in1, d_ptr->img_in2);
	std::cout << "Gamma calculation is on-going..." <<std::endl;
    d_ptr->do_gamma_analysis ();

	//compose a report string
	d_ptr->compose_report();
}

Plm_image::Pointer
Gamma_dose_comparison::get_gamma_image ()
{
    if (!d_ptr->have_gamma_image) {
        this->run();
    }
    return d_ptr->gamma_image;
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
    d_ptr->mode = PASS;
    d_ptr->do_gamma_threshold ();
    return d_ptr->labelmap_out;
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
    d_ptr->mode = FAIL;
    d_ptr->do_gamma_threshold ();
    return d_ptr->labelmap_out;
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

void 
Gamma_dose_comparison::resample_image_with_fixed_spacing (
	Plm_image *input_img, float spacing[3])
{	
	Plm_image_header pih;
	pih.set_from_plm_image (input_img);	

	long dim[3];	
	///* Auto-adjust, keep same image extent */
	float extent[3];
	pih.get_image_extent (extent);
	//std::cout << "Extent " << extent << std::endl;

	//std::cout << "extent " << extent[0] << " " << extent[1] << " " << extent[2] << std::endl;
	plm_long new_dim[3];
	for (int d = 0; d < 3; d++) {
	new_dim[d] = 1 + FLOOR_PLM_LONG (extent[d] / spacing[d]);
	}
	pih.set_spacing(spacing);
	pih.set_dim (new_dim);	

	itk::Image<float, 3>::Pointer resampledImage = resample_image (
		input_img->itk_float(),
		&pih,
		0.f,
		false
		);
	input_img->set_itk(resampledImage);
}

/* -------------------------------------------------------------------------
   Private functions
   ------------------------------------------------------------------------- */
void 
Gamma_dose_comparison_private::find_reference_max_dose ()
{
    FloatImageType::Pointer itk_1 = img_in1->itk_float();
    typedef itk::ImageRegionIteratorWithIndex< 
        FloatImageType > FloatIteratorType;
    typedef itk::ImageRegion<3> FloatRegionType;
    
    FloatRegionType all_of_img1 = itk_1->GetLargestPossibleRegion();
    FloatIteratorType itk_1_iterator (itk_1, all_of_img1);
    float maxlevel1=-1e20;
    for (itk_1_iterator.GoToBegin(); 
         !itk_1_iterator.IsAtEnd(); 
         ++itk_1_iterator)
    {
        float level1 = itk_1_iterator.Get();
        if (level1 > maxlevel1) maxlevel1 = level1;         
    } 
    this->dose_max = maxlevel1;

    lprintf ("Gamma dose max is %f\n", this->dose_max);
}

void 
Gamma_dose_comparison_private::do_gamma_analysis ()
{ 
    float spacing_in[3], origin_in[3];
    plm_long dim_in[3];
    Plm_image_header pih;
    float gamma;

    FloatImageType::Pointer itk_1 = img_in1->itk_float();
    FloatImageType::Pointer itk_2 = img_in2->itk_float();
    UCharImageType::Pointer mask_img;
    if (img_mask) {
        mask_img = img_mask->itk_uchar();
    }

    pih.set_from_itk_image (itk_1);
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
    
    FloatRegionType all_of_img1 = itk_1->GetLargestPossibleRegion();
    FloatRegionType all_of_img2 = itk_2->GetLargestPossibleRegion();
    FloatRegionType subset_of_img2;


    FloatIteratorType itk_1_iterator (itk_1, all_of_img1);
    FloatIteratorType gamma_img_iterator (gamma_img, gamma_img->GetLargestPossibleRegion());
    UCharIteratorType mask_img_iterator;
    if (mask_img) {
        mask_img_iterator = UCharIteratorType (mask_img, mask_img->GetLargestPossibleRegion());
    }

    FloatImageType::IndexType k1, k2, k3;
    FloatImageType::OffsetType offset;
    FloatImageType::SizeType region_size;
    FloatPoint3DType phys;

    //int reg_pixsize; 
    float level1, level2, dr2, dd2, gg;
    float f0,f1,f2,f3;

    // vox-to-mm-to-gamma conversion factors
    // strictly, these should come from IMAGE2, not 1
    f0 = spacing_in[0]/this->dta_tolerance; f0=f0*f0;
    f1 = spacing_in[1]/this->dta_tolerance; f1=f1*f1;
    f2 = spacing_in[2]/this->dta_tolerance; f2=f2*f2;
	//dose_difference_tolerance: e.g.: 0.03
	//hence, dose_tol [Gy]
    float dose_tol = this->reference_dose * this->dose_difference_tolerance;

    f3 = 1./dose_tol; f3 = f3*f3;
    
    // compute search region size
    float gmax_dist = this->dta_tolerance * this->gamma_max;
    offset[0] = (int) ceil (gmax_dist /fabs(spacing_in[0]));
    offset[1] = (int) ceil (gmax_dist /fabs(spacing_in[1]));
    offset[2] = (int) ceil (gmax_dist /fabs(spacing_in[2]));		

	/*have_reference_dose = false;
	reference_dose = 0.f;
	have_analysis_thresh = false;
	analysis_thresh = 0.f;*/

	//analysis_threshold in Gy
    float analysis_threshold = this->analysis_thresh * this->reference_dose;
	//default: if no option of analysis threshold is used: analysis_threshold = 0

    gamma_img_iterator.GoToBegin();
    if (mask_img) {
        mask_img_iterator.GoToBegin();
    }
	
	//This value is -1.0 in OmniproIMRT
	float NoProcessGammaValue = 0.0f;
	
	i_total_vox_num =0;
    for (itk_1_iterator.GoToBegin(); 
         !itk_1_iterator.IsAtEnd(); 
         ++itk_1_iterator)
    {
		i_total_vox_num++;
        // skip masked out voxels
        // (mask may be interpolated so we use a value of 0.5 for threshold)
        if (mask_img) {
            unsigned char mask_value = mask_img_iterator.Get();
            ++mask_img_iterator;
			//if mask value is less than 0.5, gamma value is 0.0 (passed)
            if (mask_value < 0.5) {
                gamma_img_iterator.Set (NoProcessGammaValue);//YK: is 0.0 Safe? how about -1.0?
                ++gamma_img_iterator;
                continue;
            }
        }	

        //calculate gamma for this voxel of input image
        level1 = itk_1_iterator.Get();

		//if this option is on, computation will be much faster because dose comparison will not be perforemd.
		if (this->b_skip_low_dose_gamma){
			if (this->have_analysis_thresh){
				if (level1 < analysis_threshold){
					gamma_img_iterator.Set (NoProcessGammaValue);//YK: is 0.0 Safe? how about -1.0?
					++gamma_img_iterator;
					continue;//skip the rest of the loop. This point will not be counted in analysis
				}			
			}
		}

        k1=itk_1_iterator.GetIndex();
        itk_1->TransformIndexToPhysicalPoint( k1, phys );
        itk_2->TransformPhysicalPointToIndex( phys, k2 );				

        //k2 is the voxel index of the k1's physical (mm) position in img2
    
        // set subset_of_img2 to the following region:  
        // k1[0]-region_size < k2[0] < k1[0]+region_size, same for y,z
        // assume (approx) same pix spacing in img1 and img2
        // crop the region by the entire image to be safe
        k2 -= offset;  
        subset_of_img2.SetIndex (k2);
        region_size[0] = 2 * offset[0] + 1;
        region_size[1] = 2 * offset[1] + 1;
        region_size[2] = 2 * offset[2] + 1;
        subset_of_img2.SetSize (region_size);
        subset_of_img2.Crop (all_of_img2);

        FloatIteratorType itk_2_iterator (itk_2, subset_of_img2);

        // calculate gamma, take a minimum of ... over the subset_of_img2
        gamma = 1e20;
        for (itk_2_iterator.GoToBegin(); 
             !itk_2_iterator.IsAtEnd(); 
             ++itk_2_iterator)
        {
            k3 = itk_2_iterator.GetIndex();
            level2 = itk_2_iterator.Get();
			//(k3-k1)^2: px^2, f = [1/px^2] --> dr2 = dimensionless
            dr2 = (k3[0]-k1[0])*(k3[0]-k1[0])*f0 +
                (k3[1]-k1[1])*(k3[1]-k1[1])*f1 +
                (k3[2]-k1[2])*(k3[2]-k1[2])*f2 ;
			//dd2: (dose diff./D)^2

			//if local gamma option is on, redefine f3
			if (b_local_gamma){
				if (level1 > 0){
				f3 = 1.0 / (level1 * this->dose_difference_tolerance); //dose_difference_tolerance = 0.03 in default setup
				f3 = f3*f3;
				}
			}
			///YK: what if level1 <= 0? this case will be included in the gamma pass rate calculation			
			dd2 = (level1 - level2) * (level1 - level2) * f3; // (if local gamma is on, {[(d1-d2)/d1]*100 (%) / tol_dose(%)}^2)            

            gg = dr2 + dd2;
			// in this subregion, only minimum value is take.
            if (gg < gamma) gamma=gg;
        }
        gamma = sqrt(gamma);

		//gamma_max: e.g. 2.0
        if (gamma > this->gamma_max) {
            gamma = this->gamma_max;
        }
        gamma_img_iterator.Set (gamma);
        ++gamma_img_iterator;

        /* Get statistics */
        if (this->have_analysis_thresh) {
            if (level1 > analysis_threshold) {
                this->analysis_num_vox ++;
                if (gamma <= 1) {
                    this->analysis_num_pass ++;
                }
            }
        }
		//if no analysis threshold was used: every dose point will be counted as analysis point
		else{
			this->analysis_num_vox++;
			if (gamma <= 1) {
				this->analysis_num_pass++;
			}
		}
    }//end of each voxel iterator of img1 (ref image)
    this->gamma_image->set_itk (gamma_img);	
	
}

void 
Gamma_dose_comparison_private::do_gamma_threshold ()
{ 
    FloatImageType::Pointer ref_img = img_in1->itk_float();
    FloatImageType::Pointer gamma_img = this->gamma_image->itk_float();

    /* Create labelmap image if not already created */
    if (!labelmap_out) {
        labelmap_out = new Plm_image;
        UCharImageType::Pointer gamma_labelmap = UCharImageType::New();
        itk_image_header_copy (gamma_labelmap, gamma_img);
        gamma_labelmap->Allocate();
        labelmap_out = new Plm_image (gamma_labelmap);
    }
    UCharImageType::Pointer gamma_labelmap = labelmap_out->itk_uchar();

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
        switch (mode) {
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

void 
Gamma_dose_comparison_private::do_mask_threshold ()
{ 
    UCharImageType::Pointer mask_img = img_mask->itk_uchar();

    typedef itk::ImageRegionIteratorWithIndex< UCharImageType > 
        UCharIteratorType;
    UCharIteratorType mask_it (mask_img, 
        mask_img->GetLargestPossibleRegion());

    /* Loop through mask image, threshold */
    for (mask_it.GoToBegin(); !mask_it.IsAtEnd(); ++mask_it)
    {
        unsigned char mask_val = mask_it.Get();
        mask_it.Set( mask_val<1 ? 0 : 1 );
    }
}

void Gamma_dose_comparison_private::compose_report()
{
	str_gamma_report ="";

	int iStrSize = 128;
	char itemStr[128];

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%d\n","local_gamma_on", this->b_local_gamma);
	str_gamma_report= std::string(itemStr);

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","analysis_threshold", this->analysis_thresh);	
	str_gamma_report= str_gamma_report+std::string(itemStr);
	
	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%d\n","skip_low_dose_gamma", this->b_skip_low_dose_gamma);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","gamma_max", this->gamma_max);
	str_gamma_report= str_gamma_report+std::string(itemStr);
	
	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","inherent_resample(mm)", this->f_inherent_resample_mm);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","reference_dose_Gy", this->reference_dose);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","dose_difference_tolerance", this->dose_difference_tolerance);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","dta_tolerance", this->dta_tolerance);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);
	sprintf( itemStr, "%s\t%3.2f\n","dose_max", this->dose_max);
	str_gamma_report= str_gamma_report+std::string(itemStr);


	memset(itemStr, 0, iStrSize);	
	sprintf( itemStr, "%s\t%d\n","number_of_total_voxels", this->i_total_vox_num);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);	
	sprintf( itemStr, "%s\t%d\n","number_of_analysis_voxels", this->analysis_num_vox);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);	
	sprintf( itemStr, "%s\t%d\n","number_of_pass_voxels", this->analysis_num_pass);
	str_gamma_report= str_gamma_report+std::string(itemStr);

	memset(itemStr, 0, iStrSize);	
	sprintf( itemStr, "%s\t%3.2f\n","pass_rate(%)", analysis_num_pass/(float)analysis_num_vox*100.0);
	str_gamma_report= str_gamma_report+std::string(itemStr);	
}

std::string
Gamma_dose_comparison::get_report_string()
{
	return d_ptr->str_gamma_report;
}

void
Gamma_dose_comparison::set_report_string(std::string& report_str)
{
	d_ptr->str_gamma_report = report_str;
}


bool
Gamma_dose_comparison::is_local_gamma()
{
	return d_ptr->b_local_gamma;
}

void
Gamma_dose_comparison::set_local_gamma(bool bLocalGamma)
{
	d_ptr->b_local_gamma = bLocalGamma;
}


bool
Gamma_dose_comparison::is_skip_low_dose_gamma()
{
	return d_ptr->b_skip_low_dose_gamma;
}

void
Gamma_dose_comparison::set_skip_low_dose_gamma(bool bSkipLowDoseGamma)
{
	d_ptr->b_skip_low_dose_gamma = bSkipLowDoseGamma;
}

float
Gamma_dose_comparison::get_inherent_resample_mm()
{
	return d_ptr->f_inherent_resample_mm;
}

void
Gamma_dose_comparison::set_inherent_resample_mm(float inherent_spacing_mm)
{
	d_ptr->f_inherent_resample_mm = inherent_spacing_mm;
}