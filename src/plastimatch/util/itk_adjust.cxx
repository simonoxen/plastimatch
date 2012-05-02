/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <time.h>
#include "itkImageRegionIterator.h"

#include "itk_adjust.h"
#include "plm_image.h"

//numeric_limits<float>::min(), numeric_limits<float>::max()

void
itk_adjust (FloatImageType::Pointer image, const Adjustment_list& al)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    FloatImageType::RegionType rg = image->GetLargestPossibleRegion ();
    FloatIteratorType it (image, rg);
    Adjustment_list::const_iterator ait;

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float v = it.Get();
        for (ait = al.begin(); ait != al.end(); ++ait) {
            printf ("%f %f\n", ait->first, ait->second);
            //it.Set (parms->truncate_above);
        }
        break;
    }
}
