/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImage.h"

#include "itk_image.h"
#include "plm_image_header.h"

template <class T>
UCharImage4DType::Pointer
itk_image_convert_uchar_4d (T im_in)
{
    int d;
    UCharImage4DType::Pointer im_out = UCharImage4DType::New();

    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;

    typename ImageType::RegionType rgn_in = im_in->GetLargestPossibleRegion();
    const typename ImageType::PointType& og_in = im_in->GetOrigin();
    const typename ImageType::SpacingType& sp_in = im_in->GetSpacing();

    UCharImage4DType::RegionType rgn_out;
    UCharImage4DType::PointType og_out;
    UCharImage4DType::SpacingType sp_out;
    //UCharImage4DType::RegionType::IndexType idx_out;
    UCharImage4DType::RegionType::SizeType sz_out;

    sz_out[0] = 1;
    og_out[0] = 0;
    sp_out[0] = 1;
    for (d = 0; d < 3; d++) {
	sz_out[d+1] = rgn_in.GetSize()[d];
	og_out[d+1] = og_in[d];
	sp_out[d+1] = sp_in[d];
    }
    rgn_out.SetSize (sz_out);
    im_out->SetRegions (rgn_out);
    im_out->SetOrigin (og_out);
    im_out->SetSpacing (sp_out);
    im_out->Allocate ();

    return im_out;
}


/* Explicit instantiations */
template plastimatch1_EXPORT UCharImage4DType::Pointer itk_image_convert_uchar_4d (UInt32ImageType::Pointer);
