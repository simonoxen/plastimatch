/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include "itk_pointset.h"
#include "print_and_exit.h"

/* Don't get confused by the parameterization of the itk pointset.  The 
   PixelType is the "color" of the point, whereas the PointType is the 
   type used to represent the coordinate location */

template<class T>
void
pointset_load (T pointset, char* fn)
{
    typedef typename T::ObjectType PointSetType;
    typedef typename PointSetType::PointType PointType;
    typedef typename PointSetType::PointsContainer PointsContainerType;

    FILE* fp;
    const int MAX_LINE = 2048;
    char line[MAX_LINE];
    float p[3];
    PointType tp;

    fp = fopen (fn, "r");
    if (!fp) {
	print_and_exit ("Error loading pointset file: %s\n", fn);
    }

    typename PointsContainerType::Pointer points = PointsContainerType::New();

    unsigned int i = 0;
    while (fgets (line, MAX_LINE, fp)) {
	if (sscanf (line, "%g %g %g", &p[0], &p[1], &p[2]) != 3) {
	    print_and_exit ("Warning: bogus line in pointset file \"%s\"\n", fn);
	}
	tp[0] = p[0];
	tp[1] = p[1];
	tp[2] = p[2];
	printf ("Loading: %g %g %g\n", p[0], p[1], p[2]);
	points->InsertElement (i++, tp);
    }
    pointset->SetPoints (points);

    fclose (fp);
}

template<class T>
T
pointset_warp (T ps_in, Xform* xf)
{
    typedef typename T::ObjectType PointSetType;
    typedef typename PointSetType::PointType PointType;
    typedef typename PointSetType::PixelType PixelType;
    typedef typename PointSetType::PointsContainer PointsContainerType;
    typedef typename PointsContainerType::Iterator PointsIteratorType;

    typename PointSetType::Pointer ps_out = PointSetType::New();
    typename PointsContainerType::Pointer points_out = PointsContainerType::New();
    typename PointsContainerType::Pointer points_in = ps_in->GetPoints ();
    PointType tp;

    PointsIteratorType it = points_in->Begin();
    PointsIteratorType end = points_in->End();
    unsigned int i = 0;
    while (it != end) {
	PointType p = it.Value();
        xform_transform_point (&tp, xf, p);
	points_out->InsertElement (i, tp);
	++it;
	++i;
    }
    ps_out->SetPoints (points_out);
    return ps_out;
}

template<class T>
void
pointset_debug (T pointset)
{
    typedef typename T::ObjectType PointSetType;
    typedef typename PointSetType::PointType PointType;
    typedef typename PointSetType::PointsContainer PointsContainerType;
    typedef typename PointsContainerType::Iterator PointsIteratorType;

    typename PointsContainerType::Pointer points = pointset->GetPoints ();

    PointsIteratorType it = points->Begin();
    PointsIteratorType end = points->End();
    while (it != end) {
	PointType p = it.Value();
	printf ("%g %g %g\n", p[0], p[1], p[2]);
	++it;
    }
}

/* Explicit instantiations */
template plastimatch1_EXPORT void pointset_debug (PointSetType::Pointer pointset);
template plastimatch1_EXPORT void pointset_load (PointSetType::Pointer pointset, char* fn);
template plastimatch1_EXPORT PointSetType::Pointer pointset_warp (PointSetType::Pointer ps_in, Xform* xf);
