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
#include "itkCastImageFilter.h"

#include "gamma_analysis.h"
#include "itk_image_type.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"

void find_dose_threshold( Gamma_parms *parms ) {
        
    FloatImageType::Pointer img_in1 = parms->img_in1->itk_float();
    
    typedef itk::ImageRegionIteratorWithIndex< FloatImageType > FloatIteratorType;
    typedef itk::ImageRegion<3> FloatRegionType;
    
    FloatRegionType all_of_img1 = img_in1->GetLargestPossibleRegion();
    
    FloatIteratorType img_in1_iterator (img_in1, all_of_img1);
   
    float level1, maxlevel1=-1e20;
        for (img_in1_iterator.GoToBegin(); !img_in1_iterator.IsAtEnd(); ++img_in1_iterator) {
            level1 = img_in1_iterator.Get();
            if (level1 > maxlevel1) maxlevel1 = level1;         
        } 
        parms->dose_max = maxlevel1;
}

void do_gamma_analysis( Gamma_parms *parms ) { 

    float spacing_in[3], origin_in[3];
    plm_long dim_in[3];
    Plm_image_header pih;
    float gamma;

    FloatImageType::Pointer img_in1 = parms->img_in1->itk_float();
    FloatImageType::Pointer img_in2 = parms->img_in2->itk_float();

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
    f0 = spacing_in[0]/parms->r_tol; f0=f0*f0;
    f1 = spacing_in[1]/parms->r_tol; f1=f1*f1;
    f2 = spacing_in[2]/parms->r_tol; f2=f2*f2;
    f3 = 1./parms->d_tol; f3 = f3*f3;

    // get min spacing, safeguard against negative spacings.
    float min_spc = fabs(spacing_in[0]);
    if (fabs(spacing_in[1])<min_spc) min_spc=fabs(spacing_in[1]);
    if (fabs(spacing_in[2])<min_spc) min_spc=fabs(spacing_in[2]);

    // if gamma is limited to gamma_max, 
    // no need to look beyond reg_pixsize away from the current voxel
    reg_pixsize = (int)ceil(1+  parms->gamma_max/min_spc);
    
    offset[0]=reg_pixsize;
    offset[1]=reg_pixsize;
    offset[2]=reg_pixsize;

    gamma_img_iterator.GoToBegin();
    gamma_labelmap_iterator.GoToBegin();

    for (img_in1_iterator.GoToBegin(); !img_in1_iterator.IsAtEnd(); ++img_in1_iterator) {
    
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
        k2-= offset;  
        subset_of_img2.SetIndex(k2);
        region_size.Fill( 2*reg_pixsize ); 
        subset_of_img2.SetSize( region_size);
        subset_of_img2.Crop( all_of_img2 );

        FloatIteratorType img_in2_iterator (img_in2, subset_of_img2);

        // calculate gamma, take a minimum of ... over the subset_of_img2
        gamma = 1e20;
        for (img_in2_iterator.GoToBegin(); !img_in2_iterator.IsAtEnd(); ++img_in2_iterator) {

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
        if (gamma > parms->gamma_max) gamma = parms->gamma_max;

        // test only: gamma = phys[0];

        gamma_img_iterator.Set (gamma);

        switch (parms->mode) {
        case PASS:
            if ((gamma >=0) && (gamma <= 1)) {
                /* only set label map voxel if there is dose in image 1 */
                if (level1 > 0) gamma_labelmap_iterator.Set (1);
            } else {
                gamma_labelmap_iterator.Set (0);
            }
            ++gamma_labelmap_iterator;
            break;
        case FAIL:
            if (gamma > 1) {
                gamma_labelmap_iterator.Set (1);
            } else {
                gamma_labelmap_iterator.Set (0);
            }
            ++gamma_labelmap_iterator;
            break;
        case NONE:
        default:
            break;
        }
        ++gamma_img_iterator;
    }

    parms->img_out = new Plm_image;
    parms->img_out->set_itk (gamma_img);

    if (parms->mode != NONE) {
        parms->labelmap_out = new Plm_image;
        parms->labelmap_out->set_itk (gamma_labelmap);
    }
}

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
