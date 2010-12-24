/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include "itk_pointset.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "xform_point.h"

/* Don't get confused by the parameterization of the itk pointset.  The 
   PixelType is the "color" of the point, whereas the PointType is the 
   type used to represent the coordinate location */

Pointset*
pointset_from_itk_float_pointset (FloatPointSetType::Pointer itk_ps)
{
    typedef FloatPointSetType::PointsContainer PointsContainerType;
    typedef PointsContainerType::Iterator PointsIteratorType;

    Pointset *ps = pointset_create ();
    PointsContainerType::Pointer itk_ps_c 
	= itk_ps->GetPoints ();

    PointsIteratorType it = itk_ps_c->Begin();
    PointsIteratorType end = itk_ps_c->End();
    unsigned int i = 0;
    while (it != end) {
	FloatPointType p = it.Value();
	pointset_resize (ps, i + 1);
	ps->points[i*3+0] = p[0];
	ps->points[i*3+1] = p[1];
	ps->points[i*3+2] = p[2];
	++it;
	++i;
    }
    return ps;
}

FloatPointSetType::Pointer
itk_float_pointset_from_pointset (Pointset *ps)
{
    FloatPointSetType::Pointer itk_ps = FloatPointSetType::New ();
    FloatPointSetType::PointsContainer::Pointer itk_ps_c 
	= itk_ps->GetPoints ();

    FloatPointIdType id = itk::NumericTraits< FloatPointIdType >::Zero;
    for (int i = 0; i < ps->num_points; i++) {
	FloatPointType p1;
	p1[0] = ps->points[i*3+0];
	p1[1] = ps->points[i*3+1];
	p1[2] = ps->points[i*3+2];
	itk_ps_c->InsertElement (id++, p1);
    }
    return itk_ps;
}

DoublePointSetType::Pointer
itk_double_pointset_from_pointset (Pointset *ps)
{
    DoublePointSetType::Pointer itk_ps = DoublePointSetType::New ();
    DoublePointSetType::PointsContainer::Pointer itk_ps_c 
	= itk_ps->GetPoints ();

    DoublePointIdType id = itk::NumericTraits< DoublePointIdType >::Zero;
    for (int i = 0; i < ps->num_points; i++) {
	DoublePointType p1;
	p1[0] = ps->points[i*3+0];
	p1[1] = ps->points[i*3+1];
	p1[2] = ps->points[i*3+2];
	itk_ps_c->InsertElement (id++, p1);
    }
    return itk_ps;
}

template<class T>
void
itk_pointset_load (T pointset, const char* fn)
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
itk_pointset_warp (T ps_in, Xform* xf)
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
        xform_point_transform (&tp, xf, p);
	points_out->InsertElement (i, tp);
	++it;
	++i;
    }
    ps_out->SetPoints (points_out);
    return ps_out;
}

template<class T>
void
itk_pointset_debug (T pointset)
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
template plastimatch1_EXPORT void itk_pointset_debug (FloatPointSetType::Pointer pointset);
template plastimatch1_EXPORT void itk_pointset_debug (DoublePointSetType::Pointer pointset);
template plastimatch1_EXPORT void itk_pointset_load (FloatPointSetType::Pointer pointset, const char* fn);
template plastimatch1_EXPORT FloatPointSetType::Pointer itk_pointset_warp (FloatPointSetType::Pointer ps_in, Xform* xf);
