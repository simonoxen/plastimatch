/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>
#include "itkImageRegionIterator.h"

#include "itk_adjust.h"
#include "itk_image_clone.h"
#include "plm_math.h"
#include "print_and_exit.h"

FloatImageType::Pointer
itk_adjust (FloatImageType::Pointer image_in, const Adjustment_list& al)
{
    FloatImageType::Pointer image_out = itk_image_clone (image_in);

    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
    FloatImageType::RegionType rg = image_out->GetLargestPossibleRegion ();
    FloatIteratorType it (image_out, rg);

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

    /* Debug adjustment lists */
#if defined (commentout)
    Adjustment_list::const_iterator it_d = ait_start;
    while (it_d != ait_end) {
        printf ("[%f,%f]\n", it_d->first, it_d->second);
        it_d ++;
    }
    printf ("[%f,%f]\n", it_d->first, it_d->second);
    printf ("slopes [%f,%f]\n", left_slope, right_slope);
#endif
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float vin = it.Get();
        float vout;

        /* Three possible cases: before first node, between two nodes, and 
           after last node */

        /* Case 1 */
        if (vin <= ait_start->first) {
            vout = ait_start->second + (vin - ait_start->first) * left_slope;
#if defined (commentout)
            printf ("[1] < %f (%f -> %f)\n", ait_start->first, vin, vout);
#endif
            goto found_vout;
        }
        else if (ait_start != ait_end) {
            Adjustment_list::const_iterator ait = ait_start;
            Adjustment_list::const_iterator prev = ait_start;
            ait++;
            do {
                /* Case 2 */
                if (vin <= ait->first) {
                    float slope = (ait->second - prev->second) 
                        / (ait->first - prev->first);
                    vout = prev->second + (vin - prev->first) * slope;
#if defined (commentout)
                    printf ("[2] in (%f,%f) (%f -> %f)\n", prev->first, 
                        ait->first, vin, vout);
#endif
                    goto found_vout;
                }
                prev = ait;
            } while (++ait != al.end());
        }
        /* Case 3 */
        vout = ait_end->second + (vin - ait_end->first) * right_slope;
#if defined (commentout)
        printf ("[3] > %f (%f -> %f)\n", ait_end->first, vin, vout);
#endif
    found_vout:
        it.Set (vout);
    }
    return image_out;
}

FloatImageType::Pointer
itk_adjust (FloatImageType::Pointer image_in, const std::string& adj_string)
{
    Adjustment_list al;
    const char* c = adj_string.c_str();
    bool have_curve = false;

    while (1) {
        int n;
        float f1, f2;
        int rc = sscanf (c, " %f , %f %n", &f1, &f2, &n);
        if (rc < 2) {
            break;
        }
        have_curve = true;

        /* Look for end-caps */
        if (!is_number(f1)) {
            if (al.size() == 0) {
                f1 = -std::numeric_limits<float>::max();
            } else {
                f1 = std::numeric_limits<float>::max();
            }
        }
        /* Append (x,y) pair to list */
        al.push_back (std::make_pair (f1, f2));

        /* Look for next pair in string */
        c += n;
        if (*c == ',') c++;
    }

    if (!have_curve) {
        print_and_exit ("Error: couldn't parse adjust string: %s\n",
            adj_string.c_str());
    }

    return itk_adjust (image_in, al);
}

FloatImageType::Pointer
itk_auto_adjust (FloatImageType::Pointer image_in)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    FloatImageType::RegionType rg = image_in->GetLargestPossibleRegion ();
    FloatIteratorType it (image_in, rg);
    Adjustment_list::const_iterator ait;

    /* GCS: This is just something for spark, works for CT image differencing
       -- make a better method later */
    Adjustment_list al;
    al.push_back (std::make_pair (-std::numeric_limits<float>::max(), 0.0));
    al.push_back (std::make_pair (-200.0,0));
    al.push_back (std::make_pair (0.0,127.5));
    al.push_back (std::make_pair (+200.0,255));
    al.push_back (std::make_pair (std::numeric_limits<float>::max(), 0.0));

    return itk_adjust (image_in, al);
}
