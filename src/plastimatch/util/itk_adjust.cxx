/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>
#include "itkImageRegionIterator.h"

#include "plmutil.h"

//numeric_limits<float>::min(), numeric_limits<float>::max()

void
itk_adjust (FloatImageType::Pointer image, const Adjustment_list& al)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    FloatImageType::RegionType rg = image->GetLargestPossibleRegion ();
    FloatIteratorType it (image, rg);

    /* Special processing for end caps */
    float left_slope = 1.0;
    float right_slope = 1.0;
    Adjustment_list::const_iterator ait_start = al.begin();
    Adjustment_list::const_iterator ait_end = al.end();
    if (ait_start->first == -std::numeric_limits<float>::max()) {
        left_slope = ait_start->second;
        ait_start++;
    }
    if ((--ait_end)->first == std::numeric_limits<float>::max()) {
        right_slope = ait_end->second;
    }

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float vin = it.Get();
        float vout;
        /* Three possible cases: before first node, between two nodes, and 
           after last node */

        /* Case 1 */
        if (vin < ait_start->first) {
            vout = ait_start->second + (vin - ait_start->first) * left_slope;
#if defined (commentout)
            printf ("[1] < %f (%f -> %f)\n", ait_start->first, vin, vout);
#endif
            goto found_vout;
        }
        else {
            Adjustment_list::const_iterator ait = ait_start;
            Adjustment_list::const_iterator prev = ait_start;
            while (++ait != ait_end) {
                /* Case 2 */
                if (vin > prev->first) {
                    float slope = (ait->second - prev->second) 
                        / (ait->first - prev->first);
                    vout = prev->second + (vin - prev->first) * slope;
#if defined (commentout)
                    printf ("[2] in (%f,%f) (%f -> %f)\n", prev->first, 
                        ait->first, vin, vout);
#endif
                    goto found_vout;
                }
            }
        }
        /* Case 3 */
        vout = ait_end->second + (vin - ait_end->first) * right_slope;
#if defined (commentout)
        printf ("[3] > %f (%f -> %f)\n", ait_end->first, vin, vout);
#endif
    found_vout:
        it.Set (vout);
    }
}

void
itk_auto_adjust (FloatImageType::Pointer image)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    FloatImageType::RegionType rg = image->GetLargestPossibleRegion ();
    FloatIteratorType it (image, rg);
    Adjustment_list::const_iterator ait;

    /* GCS: This is just something for spark, works for CT image differencing
       -- make a better method later */
    Adjustment_list al;
    al.push_back (std::make_pair (-std::numeric_limits<float>::max(), 0.0));
    al.push_back (std::make_pair (-200.0,0));
    al.push_back (std::make_pair (0.0,127.5));
    al.push_back (std::make_pair (+200.0,255));
    al.push_back (std::make_pair (std::numeric_limits<float>::max(), 0.0));

    itk_adjust (image, al);
}
