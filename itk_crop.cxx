/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkExtractImageFilter.h"
#include "itkImage.h"

#include "itk_image.h"
#include "itk_crop.h"

template <class T>
T
itk_crop (
    T& image, 
    int new_size[6])
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::ExtractImageFilter < ImageType, ImageType > FilterType;

    typename FilterType::Pointer filter = FilterType::New();
    typename ImageType::IndexType  extract_index;
    typename ImageType::SizeType   extract_size;
    typename ImageType::RegionType extract_region;

    for (int d = 0; d < 3; d++) {
	extract_index[d] = new_size[d*2];
	extract_size[d] = new_size[d*2+1] - new_size[d*2] + 1;
    }

    extract_region.SetSize (extract_size);
    extract_region.SetIndex (extract_index);

    filter->SetInput (image);
    filter->SetExtractionRegion (extract_region);

    try {
	//filter->Update();
	filter->UpdateLargestPossibleRegion ();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception running itkExtractImageFilter.\n");
	std::cout << ex << std::endl;
	getchar();
	exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}


/* Explicit instantiations */
template plastimatch1_EXPORT UCharImageType::Pointer itk_crop (UCharImageType::Pointer&, int*);
template plastimatch1_EXPORT ShortImageType::Pointer itk_crop (ShortImageType::Pointer&, int*);
template plastimatch1_EXPORT UShortImageType::Pointer itk_crop (UShortImageType::Pointer&, int*);
template plastimatch1_EXPORT UInt32ImageType::Pointer itk_crop (UInt32ImageType::Pointer&, int*);
template plastimatch1_EXPORT FloatImageType::Pointer itk_crop (FloatImageType::Pointer&, int*);
