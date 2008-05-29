/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "itkArray.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineResampleImageFunction.h"
#include "xform.h"
#include "rad_registration.h"
#include "resample_mha.h"
#include "print_and_exit.h"
#include "volume.h"
#include "bspline.h"
#include "readmha.h"

static void alloc_itk_bsp_parms (Xform *xf, BsplineTransformType::Pointer bsp);
static DeformationFieldType::Pointer xform_gpuit_vf_to_itk_vf (Volume* vf, FloatImageType::Pointer image);

//#define USE_OLD_ITK_BSPLINES 1

int
strcmp_alt (const char* s1, const char* s2)
{
  return strncmp (s1, s2, strlen(s2));
}

int
get_parms (FILE* fp, itk::Array<double>* parms, int num_parms)
{
    float f;
    int r, s;

    s = 0;
    while (r = fscanf (fp, "%f",&f)) {
	(*parms)[s++] = (double) f;
	if (s == num_parms) break;
	if (!fp) break;
    }
    return s;
}

void
load_xform (Xform *xf, char* fn)
{
    char buf[1024];
    FILE* fp;

    fp = fopen (fn, "r");
    if (!fp) {
	print_and_exit ("Error: xf_in file %s not found\n", fn);
    }
    if (!fgets(buf,1024,fp)) {
	print_and_exit ("Error reading from xf_in file.\n");
    }

    if (strcmp_alt(buf,"ObjectType = MGH_XFORM_TRANSLATION")==0) {
	TranslationTransformType::Pointer trn = TranslationTransformType::New();
	TranslationTransformType::ParametersType xfp(12);
	int num_parms;

	num_parms = get_parms (fp, &xfp, 3);
	if (num_parms != 3) {
	    print_and_exit ("Wrong number of parameters in xf_in file.\n");
	} else {
	    trn->SetParameters(xfp);
#if defined (commentout)
	    std::cout << "Initial translation parms = " << trn << std::endl;
#endif
	}
	xf->set_trn (trn);
	fclose (fp);
    } else if (strcmp_alt(buf,"ObjectType = MGH_XFORM_VERSOR")==0) {
	VersorTransformType::Pointer vrs = VersorTransformType::New();
	VersorTransformType::ParametersType xfp(6);
	int num_parms;

	num_parms = get_parms (fp, &xfp, 6);
	if (num_parms != 6) {
	    print_and_exit ("Wrong number of parameters in xf_in file.\n");
	} else {
	    vrs->SetParameters(xfp);
#if defined (commentout)
	    std::cout << "Initial versor parms = " << vrs << std::endl;
#endif
	}
	xf->set_vrs (vrs);
	fclose (fp);
    } else if (strcmp_alt(buf,"ObjectType = MGH_XFORM_AFFINE")==0) {
	AffineTransformType::Pointer aff = AffineTransformType::New();
	AffineTransformType::ParametersType xfp(12);
	int num_parms;

	num_parms = get_parms (fp, &xfp, 12);
	if (num_parms != 12) {
	    print_and_exit ("Wrong number of parameters in xf_in file.\n");
	} else {
	    aff->SetParameters(xfp);
#if defined (commentout)
	    std::cout << "Initial affine parms = " << aff << std::endl;
#endif
	}
	xf->set_aff (aff);
	fclose (fp);
    } else if (strcmp_alt(buf,"ObjectType = MGH_XFORM_BSPLINE")==0) {
	int s[3];
	float p[3];
	BsplineTransformType::RegionType::SizeType size;
	BsplineTransformType::RegionType region;
	BsplineTransformType::SpacingType spacing;
	BsplineTransformType::OriginType origin;
	BsplineTransformType::Pointer bsp = BsplineTransformType::New();

	/* Skip 2 lines */
	fgets(buf,1024,fp);
	fgets(buf,1024,fp);

	/* Load bulk transform, if it exists */
	fgets(buf,1024,fp);
	if (!strncmp ("BulkTransform", buf, strlen("BulkTransform"))) {
	    TranslationTransformType::Pointer trn = TranslationTransformType::New();
	    VersorTransformType::Pointer vrs = VersorTransformType::New();
	    AffineTransformType::Pointer aff = AffineTransformType::New();
	    itk::Array <double> xfp(12);
	    float f;
	    int n, num_parm = 0;
	    char *p = buf + strlen("BulkTransform = ");
	    while (sscanf (p, " %g%n", &f, &n) > 0) {
		if (num_parm>=12) {
		    print_and_exit ("Error loading bulk transform\n");
		}
		xfp[num_parm] = f;
		p += n;
		num_parm++;
	    }
	    if (num_parm == 12) {
		aff->SetParameters(xfp);
#if defined (commentout)
		std::cout << "Bulk affine = " << aff;
#endif
		bsp->SetBulkTransform (aff);
	    } else if (num_parm == 6) {
		vrs->SetParameters(xfp);
#if defined (commentout)
		std::cout << "Bulk versor = " << vrs;
#endif
		bsp->SetBulkTransform (vrs);
	    } else if (num_parm == 3) {
		trn->SetParameters(xfp);
#if defined (commentout)
		std::cout << "Bulk translation = " << trn;
#endif
		bsp->SetBulkTransform (trn);
	    } else {
		print_and_exit ("Error loading bulk transform\n");
	    }
	    fgets(buf,1024,fp);
	}

	/* Load origin, spacing, size */
	if (3 != sscanf(buf,"Offset = %g %g %g",&p[0],&p[1],&p[2])) {
	    print_and_exit ("Unexpected line in xform_in file.\n");
	}
	origin[0] = p[0]; origin[1] = p[1]; origin[2] = p[2];

	fgets(buf,1024,fp);
	if (3 != sscanf(buf,"ElementSpacing = %g %g %g",&p[0],&p[1],&p[2])) {
	    print_and_exit ("Unexpected line in xform_in file.\n");
	}
	spacing[0] = p[0]; spacing[1] = p[1]; spacing[2] = p[2];

	fgets(buf,1024,fp);
	if (3 != sscanf(buf,"DimSize = %d %d %d",&s[0],&s[1],&s[2])) {
	    print_and_exit ("Unexpected line in xform_in file.\n");
	}
	size[0] = s[0]; size[1] = s[1]; size[2] = s[2];

#if defined (commentout)
	std::cout << "Offset = " << origin << std::endl;
	std::cout << "Spacing = " << spacing << std::endl;
	std::cout << "Size = " << size << std::endl;
#endif

	fgets(buf,1024,fp);
	if (strcmp_alt (buf, "ElementDataFile = LOCAL")) {
	    print_and_exit ("Error: bspline xf_in failed sanity check\n");
	}

	region.SetSize (size);
	bsp->SetGridSpacing (spacing);
	bsp->SetGridOrigin (origin);
	bsp->SetGridRegion (region);

	/* Allocate memory, and bind to bspline struct to xform struct */
	alloc_itk_bsp_parms (xf, bsp);

	/* Read bspline coefficients */
	/* GCS WARNING: I'm assuming this is OK after SetParameters() */
	const unsigned int num_parms = bsp->GetNumberOfParameters();
	for (int i = 0; i < num_parms; i++) {
	    float d;
	    if (!fgets(buf,1024,fp)) {
		print_and_exit ("Missing bspline coefficient from xform_in file.\n");
	    }
	    if (1 != sscanf(buf,"%g",&d)) {
		print_and_exit ("Bad bspline parm in xform_in file.\n");
	    }
	    xf->m_itk_bsp_parms[i] = d;
	}
	fclose (fp);

	/* Linux version seems to require this... */
	bsp->SetParameters (xf->m_itk_bsp_parms);

    } else {
	/* Close the file and try again, it is probably a vector field */
	fclose (fp);
	DeformationFieldType::Pointer vf = DeformationFieldType::New();
	vf = load_float_field (fn);
	if (!vf) {
	    print_and_exit ("Unexpected file format for xf_in file.\n");
	}
	xf->set_itk_vf (vf);
    }
}

void
save_xform_translation (TranslationTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,"ObjectType = MGH_XFORM_TRANSLATION\n");
    for (int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
save_xform_versor (VersorTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,"ObjectType = MGH_XFORM_VERSOR\n");
    for (int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
save_xform_affine (AffineTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,"ObjectType = MGH_XFORM_AFFINE\n");
    for (int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
save_xform_itk_bsp (BsplineTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,
	"ObjectType = MGH_XFORM_BSPLINE\n"
	"NDims = 3\n"
	"BinaryData = False\n");
    if (transform->GetBulkTransform()) {
		if ((!strcmp("TranslationTransform", transform->GetBulkTransform()->GetNameOfClass())) ||
			(!strcmp("AffineTransform", transform->GetBulkTransform()->GetNameOfClass())) ||
			(!strcmp("VersorTransform", transform->GetBulkTransform()->GetNameOfClass())) ||
			(!strcmp("VersorRigid3DTransform", transform->GetBulkTransform()->GetNameOfClass()))) {
			fprintf (fp, "BulkTransform =");
			for (int i = 0; i < transform->GetBulkTransform()->GetNumberOfParameters(); i++) {
				fprintf (fp, " %g", transform->GetBulkTransform()->GetParameters()[i]);
			}
			fprintf (fp, "\n");
		} else if (strcmp("IdentityTransform", transform->GetBulkTransform()->GetNameOfClass())) {
			printf("Warning!!! BulkTransform exists. Type=%s\n", transform->GetBulkTransform()->GetNameOfClass());
			printf(" # of parameters=%d\n", transform->GetBulkTransform()->GetNumberOfParameters());
			printf(" The code currently does not know how to handle this type and will not write the parameters out!\n");
		}
    }

    fprintf (fp,
	"Offset = %f %f %f\n"
	"ElementSpacing = %f %f %f\n"
	"DimSize = %d %d %d\n"
	"ElementDataFile = LOCAL\n",
	transform->GetGridOrigin()[0],
	transform->GetGridOrigin()[1],
	transform->GetGridOrigin()[2],
	transform->GetGridSpacing()[0],
	transform->GetGridSpacing()[1],
	transform->GetGridSpacing()[2],
	transform->GetGridRegion().GetSize()[0],
	transform->GetGridRegion().GetSize()[1],
	transform->GetGridRegion().GetSize()[2]
	);
    for (int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
save_xform (Xform *xf, char* fn)
{
    switch (xf->m_type) {
    case XFORM_ITK_TRANSLATION:
	save_xform_translation (xf->get_trn(), fn);
	break;
    case XFORM_ITK_VERSOR:
	save_xform_versor (xf->get_vrs(), fn);
	break;
    case XFORM_ITK_AFFINE:
	save_xform_affine (xf->get_aff(), fn);
	break;
    case XFORM_ITK_BSPLINE:
	save_xform_itk_bsp (xf->get_bsp(), fn);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	save_image (xf->get_itk_vf(), fn);
	break;
    case XFORM_GPUIT_BSPLINE:
	write_bspd (fn, &xf->get_gpuit_bsp()->parms);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	write_mha (fn, xf->get_gpuit_vf());
	break;
    }
}

#if defined (GCS_REARRANGING_STUFF)
void
init_versor_moments_old (RegistrationType::Pointer registration)
{
    typedef itk::CenteredTransformInitializer < VersorTransformType,
	FloatImageType, FloatImageType > TransformInitializerType;
    TransformInitializerType::Pointer initializer =
	TransformInitializerType::New();
    
    typedef VersorTransformType* VTPointer;
    VTPointer transform = static_cast<VTPointer>(registration->GetTransform());

    initializer->SetTransform(transform);
    initializer->SetFixedImage(registration->GetFixedImage());
    initializer->SetMovingImage(registration->GetMovingImage());

    initializer->GeometryOn();

    printf ("Calling Initialize Transform\n");
    initializer->InitializeTransform();

    std::cout << "Transform is " << registration->GetTransform()->GetParameters() << std::endl;
}

void
init_versor_moments (RegistrationType::Pointer registration,
		       VersorTransformType* versor)
{
    typedef itk::CenteredTransformInitializer < VersorTransformType,
	FloatImageType, FloatImageType > TransformInitializerType;
    TransformInitializerType::Pointer initializer =
	TransformInitializerType::New();

    initializer->SetTransform(versor);
    initializer->SetFixedImage(registration->GetFixedImage());
    initializer->SetMovingImage(registration->GetMovingImage());

    initializer->GeometryOn();

    printf ("Calling Initialize Transform\n");
    initializer->InitializeTransform();

    std::cout << "Transform is " << registration->GetTransform()->GetParameters() << std::endl;
}

void
set_transform_translation (RegistrationType::Pointer registration,
			Registration_Parms* regp)
{
    TranslationTransformType::Pointer transform = TranslationTransformType::New();
    TranslationTransformType::ParametersType vt(3);
    registration->SetTransform (transform);

    switch (regp->init_type) {
    case XFORM_NONE:
		{
			VersorTransformType::Pointer v = VersorTransformType::New();
			FloatVectorType dis;
			init_versor_moments (registration, v);
			dis = v->GetOffset();
			vt[0] = dis[0]; vt[1] = dis[1]; vt[2] = dis[2];
			transform->SetParameters(vt);
			std::cout << "Initial translation parms = " << transform << std::endl;
		}
		break;
    case XFORM_TRANSLATION:
		vt[0] = regp->init[0];
		vt[1] = regp->init[1];
		vt[2] = regp->init[2];
		transform->SetParameters(vt);
		break;
    case XFORM_VERSOR:
    case XFORM_AFFINE:
    case XFORM_BSPLINE:
    case XFORM_BSPLINE_ALIGNED:
    case XFORM_FROM_FILE:
    default:
		not_implemented();
		break;
    }
}

void
set_transform_versor (RegistrationType::Pointer registration,
			Registration_Parms* regp)
{
    VersorTransformType::Pointer transform = VersorTransformType::New();
    VersorTransformType::ParametersType vt(6);
    registration->SetTransform (transform);
    switch (regp->init_type) {
    case XFORM_NONE:
	init_versor_moments (registration, transform);
	break;
    case XFORM_TRANSLATION:
	not_implemented();
	break;
    case XFORM_VERSOR:
	vt[0] = regp->init[0];
	vt[1] = regp->init[1];
	vt[2] = regp->init[2];
	vt[3] = regp->init[3];
	vt[4] = regp->init[4];
	vt[5] = regp->init[5];
	transform->SetParameters(vt);
	break;
    case XFORM_AFFINE:
    case XFORM_BSPLINE:
    case XFORM_BSPLINE_ALIGNED:
    case XFORM_FROM_FILE:
    default:
	not_implemented();
	break;
    }
}

void
set_transform_affine (RegistrationType::Pointer registration,
		      Registration_Parms* regp)
{
    AffineTransformType::Pointer transform = AffineTransformType::New();
    AffineTransformType::ParametersType vt(12);
    registration->SetTransform (transform);
    switch (regp->init_type) {
    case XFORM_NONE:
	{
	    VersorTransformType::Pointer v = VersorTransformType::New();
	    init_versor_moments (registration, v);
	    transform->SetMatrix(v->GetRotationMatrix());
	    transform->SetOffset(v->GetOffset());
	    std::cout << "Initial affine parms = " << transform << std::endl;
	}
	break;
    case XFORM_TRANSLATION:
	not_implemented();
	break;
    case XFORM_VERSOR:
	{
	    VersorTransformType::Pointer v = VersorTransformType::New();
	    VersorTransformType::ParametersType vt(6);
	    vt[0] = regp->init[0];
	    vt[1] = regp->init[1];
	    vt[2] = regp->init[2];
	    vt[3] = regp->init[3];
	    vt[4] = regp->init[4];
	    vt[5] = regp->init[5];
	    v->SetParameters(vt);
	    std::cout << "Initial versor parms = " << v << std::endl;
	    transform->SetMatrix(v->GetRotationMatrix());
	    transform->SetOffset(v->GetOffset());
	    std::cout << "Initial affine parms = " << transform << std::endl;
	}
	break;
    case XFORM_AFFINE:
	{
	    AffineTransformType::ParametersType at(12);
	    for (int i=0; i<12; i++) {
		at[i] = regp->init[i];
	    }
	    transform->SetParameters(at);
	    std::cout << "Initial affine parms = " << transform << std::endl;
	}
	break;
    case XFORM_BSPLINE:
    case XFORM_BSPLINE_ALIGNED:
    case XFORM_FROM_FILE:
    default:
	not_implemented();
	break;
    }
}
#endif /* GCS_REARRANGING_STUFF */

/* -----------------------------------------------------------------------
   Conversion to itk_any
   ----------------------------------------------------------------------- */
void
init_translation_default (Xform *xf_out, Xform* xf_in, 
		      Stage_Parms* stage,
		      const OriginType& img_origin, 
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    TranslationTransformType::Pointer trn = TranslationTransformType::New();
    xf_out->set_trn (trn);
}

void
init_versor_default (Xform *xf_out, Xform* xf_in, 
		      Stage_Parms* stage,
		      const OriginType& img_origin, 
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    VersorTransformType::Pointer vrs = VersorTransformType::New();
    xf_out->set_vrs (vrs);
}

void
init_affine_default (Xform *xf_out, Xform* xf_in, 
		      Stage_Parms* stage,
		      const OriginType& img_origin, 
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    AffineTransformType::Pointer aff = AffineTransformType::New();
    xf_out->set_aff (aff);
}

void
xform_trn_to_aff (Xform *xf_out, Xform* xf_in,
		      Stage_Parms* stage,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    init_affine_default (xf_out, xf_in, stage, 
			    img_origin, img_spacing, img_region);
    xf_out->get_aff()->SetOffset(xf_in->get_trn()->GetOffset());
}

void
xform_vrs_to_aff (Xform *xf_out, Xform* xf_in,
		      Stage_Parms* stage,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    init_affine_default (xf_out, xf_in, stage, 
			    img_origin, img_spacing, img_region);
    xf_out->get_aff()->SetMatrix(xf_in->get_vrs()->GetRotationMatrix());
    xf_out->get_aff()->SetOffset(xf_in->get_vrs()->GetOffset());
}

/* -----------------------------------------------------------------------
   Conversion to itk_bsp
   ----------------------------------------------------------------------- */
static void
init_itk_bsp_region (BsplineTransformType::Pointer bsp,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region,
		      float* grid_spac)
{
    BsplineTransformType::OriginType origin = img_origin;
    BsplineTransformType::SpacingType spacing = img_spacing;
    BsplineTransformType::RegionType region;
    BsplineTransformType::RegionType::SizeType grid_size;

    ImageRegionType::SizeType img_size = img_region.GetSize();
    ImageRegionType::IndexType img_index = img_region.GetIndex();

    /* grid_method == 0 doesn't work any more.  This will need to be 
       rewritten if needed. */
#if defined (commentout)
    if (stage->grid_method == 0) {
	/* User requested specific number of control points */
	for (int i=0; i<3; i++) grid_size[i] = stage->num_grid[i];
	/* There should be 3 extra grid points for an 
	   order 3 bspline: 1 lower & 2 upper.  */
	for (int i=0; i<3; i++) grid_size[i] += 3;
	region.SetSize (grid_size);

	for (unsigned int r=0; r<Dimension; r++) {
	    origin[r] += img_index[r] * spacing[r];
	    spacing[r] *= static_cast<double>(img_size[r] - 1) / 
		    static_cast<double>(grid_size[r] - 1);
	    origin[r] -= spacing[r];
	}
	bsp->SetGridSpacing (spacing);
	bsp->SetGridOrigin (origin);
	bsp->SetGridRegion (region);
    }
#endif

    /* User requested grid spacing in mm */

    /* GCS Mar 27, 2008.  This is the old algorithm, which would  
       request too large of a grid, causing things to slow down. */
#if (USE_OLD_ITK_BSPLINES)
    BsplineTransformType::OriginType fraction_size;
    for (int i=0; i<3; i++) {
	grid_size[i] = (int) (img_size[i]*spacing[i]/stage->grid_spac[i]);
	fraction_size[i] = img_size[i]*spacing[i] - grid_size[i]*stage->grid_spac[i];
	grid_size[i] = grid_size[i] + 2;
    }
    for (int i=0; i<3; i++) grid_size[i] += 3;

#else

    /* GCS Mar 27, 2008.  This is the new algorithm. */
    BsplineTransformType::OriginType fraction_size;
    for (int i=0; i<3; i++) {
	double i1 = (img_size[i]-1)*spacing[i];
	double i2 = img_size[i]*spacing[i];
	grid_size[i] = 1 + (int) floor (i1/grid_spac[i]);
	grid_size[i] = grid_size[i] + 2;

	/* The above grid size is correct (I checked with Excel).
	   But it crashes on Linux unless I add one more CP.
	   Probably it's a bug in ITK. */
	grid_size[i] = grid_size[i] + 1;
    }
#endif

    region.SetSize (grid_size);

#if (USE_OLD_ITK_BSPLINES)
    /* fraction_size no longer used */
    for (unsigned int r=0; r<Dimension; r++)
	origin[r] += img_index[r]*spacing[r]-fraction_size[r]/2-grid_spac[r];
#else
    for (unsigned int r=0; r<Dimension; r++)
	origin[r] += img_index[r] * spacing[r] - grid_spac[r];
#endif
    for (unsigned int r=0; r<Dimension; r++)
	spacing[r] = grid_spac[r];

    bsp->SetGridSpacing (spacing);
    bsp->SetGridOrigin (origin);
    bsp->SetGridRegion (region);

    std::cout << "BSpline Region = "
	      << region;
    std::cout << "BSpline Grid Origin = "
	      << bsp->GetGridOrigin()
	      << std::endl;
    std::cout << "BSpline Grid Spacing = "
	      << bsp->GetGridSpacing()
	      << std::endl;
    std::cout << "Image Origin = "
	      << img_origin
	      << std::endl;
    std::cout << "Image Spacing = "
	      << img_spacing
	      << std::endl;
    std::cout << "Image Index = "
	      << img_index
	      << std::endl;
    std::cout << "Image Size = "
	      << img_size
	      << std::endl;
}

static void
alloc_itk_bsp_parms (Xform *xf, BsplineTransformType::Pointer bsp)
{
    const unsigned int num_parms = bsp->GetNumberOfParameters();
    if (num_parms != xf->m_itk_bsp_parms.GetSize()) {
	xf->m_itk_bsp_parms.SetSize (num_parms);
	xf->m_itk_bsp_parms.Fill (0.0);
	bsp->SetParameters (xf->m_itk_bsp_parms);
    }
#if defined (commentout)
    printf ("Bspline transform has %d free parameters\n", num_parms);
#endif
    xf->set_itk_bsp (bsp);
}

static void
init_itk_bsp_default (Xform *xf_out, Xform* xf_in, 
		      Stage_Parms* stage,
		      const OriginType& img_origin, 
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    BsplineTransformType::Pointer bsp = BsplineTransformType::New();
    init_itk_bsp_region (bsp, img_origin, img_spacing, img_region, stage->grid_spac);
    alloc_itk_bsp_parms (xf_out, bsp);
}

static void
xform_trn_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      Stage_Parms* stage,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    init_itk_bsp_default (xf_out, xf_in, stage, 
			    img_origin, img_spacing, img_region);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_trn());
}

static void
xform_vrs_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      Stage_Parms* stage,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    init_itk_bsp_default (xf_out, xf_in, stage, 
			    img_origin, img_spacing, img_region);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_vrs());
}

static void
xform_aff_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      Stage_Parms* stage,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region)
{
    init_itk_bsp_default (xf_out, xf_in, stage, 
			    img_origin, img_spacing, img_region);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_aff());
}

static void
xform_itk_bsp_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region,
		      float* grid_spac)
{
    BsplineTransformType::Pointer bsp_new = BsplineTransformType::New();
    printf ("Initializing **bsp_new**\n");
    init_itk_bsp_region (bsp_new, img_origin, img_spacing, img_region, grid_spac);
    alloc_itk_bsp_parms (xf_out, bsp_new);
    BsplineTransformType::Pointer bsp_old = xf_in->get_bsp();

    /* Need to copy the bulk transform */
    bsp_new->SetBulkTransform (bsp_old->GetBulkTransform());

    /* GCS May 12, 2008.  I feel like the below algorithm suggested 
	by ITK is wrong.  If BSplineResampleImageFunction interpolates the 
	coefficient image, the resulting resampled B-spline will be smoother 
	than the original.  But maybe the algorithm is OK, just a problem with 
	the lack of proper ITK documentation.  Need to test this. */

    // Now we need to initialize the BSpline coefficients of the higher 
    // resolution transform. This is done by first computing the actual 
    // deformation field at the higher resolution from the lower 
    // resolution BSpline coefficients. Then a BSpline decomposition 
    // is done to obtain the BSpline coefficient of the higher 
    // resolution transform.
    unsigned int counter = 0;
    for (unsigned int k = 0; k < Dimension; k++) {
	typedef BsplineTransformType::ImageType ParametersImageType;
	typedef itk::ResampleImageFilter<ParametersImageType,
		ParametersImageType> ResamplerType;
	ResamplerType::Pointer upsampler = ResamplerType::New();

	typedef itk::BSplineResampleImageFunction<ParametersImageType,
		double> FunctionType;
	FunctionType::Pointer fptr = FunctionType::New();

	typedef itk::IdentityTransform<double,
		Dimension> IdentityTransformType;
	IdentityTransformType::Pointer identity = IdentityTransformType::New();

	upsampler->SetInput (bsp_old->GetCoefficientImage()[k]);
	upsampler->SetInterpolator (fptr);
	upsampler->SetTransform (identity);
	upsampler->SetSize (bsp_new->GetGridRegion().GetSize());
	upsampler->SetOutputSpacing (bsp_new->GetGridSpacing());
	upsampler->SetOutputOrigin (bsp_new->GetGridOrigin());

	typedef itk::BSplineDecompositionImageFilter<ParametersImageType,
		ParametersImageType> DecompositionType;
	DecompositionType::Pointer decomposition = DecompositionType::New();

	decomposition->SetSplineOrder (SplineOrder);
	decomposition->SetInput (upsampler->GetOutput());
	decomposition->Update();

	ParametersImageType::Pointer newCoefficients 
		= decomposition->GetOutput();

	// copy the coefficients into the parameter array
	typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
	Iterator it (newCoefficients, bsp_new->GetGridRegion());
	while (!it.IsAtEnd()) {
	    xf_out->m_itk_bsp_parms[counter++] = it.Get();
	    ++it;
	}
    }
}

static void
gpuit_bsp_region_to_itk_bsp_region (OriginType& img_origin, 
	    SpacingType& img_spacing, ImageRegionType& img_region, 
	    int* roi_offset, int* roi_dim, 
	    float* img_offset, float* pix_spacing)
{
    ImageRegionType::SizeType img_size;
    ImageRegionType::IndexType img_index;

    /* Convert old gpuit_bsp to itk_bsp (at new grid_spac, img sampling) */
    for (int d = 0; d < Dimension; d++) {
	img_origin[d] = img_offset[d];
	img_spacing[d] = pix_spacing[d];
	img_index[d] = roi_offset[d];
	img_size[d] = roi_dim[d];
    }
    img_region.SetSize (img_size);
    img_region.SetIndex (img_index);
}

void
xform_gpuit_bsp_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region, 
		      float* grid_spac)
{
    typedef BsplineTransformType::ImageType ParametersImageType;
    typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
    Xform_GPUIT_Bspline* xgb = xf_in->get_gpuit_bsp();
    BSPLINE_Parms* parms = &xgb->parms;
    BSPLINE_Data* bspd = &parms->bspd;
    Xform xf_tmp;
    OriginType img_origin_old;
    SpacingType img_spacing_old;
    ImageRegionType img_region_old;

    /* Convert geometry info from C arrays to ITK classes */
    gpuit_bsp_region_to_itk_bsp_region (img_origin_old, img_spacing_old, img_region_old,
	    parms->roi_offset, parms->roi_dim, xgb->img_origin, xgb->img_spacing);

    /* First, create an ITK image to hold the deformation values at 
	new resolution of control point grid */
    BsplineTransformType::Pointer bsp_old = BsplineTransformType::New();
    printf ("Initializing **bsp_old**\n");
    init_itk_bsp_region (bsp_old, img_origin_old, img_spacing_old, 
			img_region_old, xgb->grid_spac);
    alloc_itk_bsp_parms (&xf_tmp, bsp_old);

    /* RMK: bulk transform is Identity (not supported by GPUIT) */

    /* Copy from GPUIT coefficient array to ITK coefficient array */
    int k = 0;
    for (int d = 0; d < Dimension; d++) {
	for (int i = 0; i < bspd->num_knots; i++) {
	    xf_tmp.m_itk_bsp_parms[k] = bspd->coeff[3*i+d] * img_spacing[d];
	    k++;
	}
    }

    /* Then, resample the xform to the desired grid spacing */
    xform_itk_bsp_to_itk_bsp (xf_out, &xf_tmp,
		      img_origin, img_spacing, img_region, grid_spac);
}

/* -----------------------------------------------------------------------
   Conversion to itk_vf
   ----------------------------------------------------------------------- */
static DeformationFieldType::Pointer
xform_any_to_itk_vf (itk::Transform<double,3,3>* xf,
		     const int* dim, float* offset, float* spacing)
{
    int i;
    DeformationFieldType::SizeType sz;
    DeformationFieldType::IndexType st;
    DeformationFieldType::RegionType rg;
    DeformationFieldType::PointType og;
    DeformationFieldType::SpacingType sp;
    DeformationFieldType::Pointer itk_vf = DeformationFieldType::New();

    /* Copy header & allocate data for itk */
    for (i = 0; i < 3; i++) {
	st[i] = 0;
	sz[i] = dim[i];
	sp[i] = spacing[i];
	og[i] = offset[i];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);

    itk_vf->SetRegions (rg);
    itk_vf->SetOrigin (og);
    itk_vf->SetSpacing (sp);
    itk_vf->Allocate ();

    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (itk_vf, itk_vf->GetBufferedRegion());

    fi.GoToBegin();

    DoublePointType fixed_point;
    DoublePointType moving_point;
    DeformationFieldType::IndexType index;

    FloatVectorType displacement;

    while (!fi.IsAtEnd()) {
	index = fi.GetIndex();
	itk_vf->TransformIndexToPhysicalPoint (index, fixed_point);
	moving_point = xf->TransformPoint (fixed_point);
	for (int r = 0; r < Dimension; r++) {
	    displacement[r] = moving_point[r] - fixed_point[r];
	}
	fi.Set (displacement);
	++fi;
    }
    return itk_vf;
}

/* Note: function name is overloaded for backward compatibility.  
   Try to rename soon... */
DeformationFieldType::Pointer
xform_any_to_itk_vf (itk::Transform<double,3,3>* xf,
		 FloatImageType::Pointer image)
{
    DeformationFieldType::Pointer field = DeformationFieldType::New();
    field->SetRegions (image->GetBufferedRegion());
    field->SetOrigin (image->GetOrigin());
    field->SetSpacing (image->GetSpacing());
    field->Allocate();

    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (field, image->GetBufferedRegion());

    fi.GoToBegin();

    DoublePointType fixed_point;
    DoublePointType moving_point;
    DeformationFieldType::IndexType index;

    FloatVectorType displacement;

    while (!fi.IsAtEnd()) {
	index = fi.GetIndex();
	field->TransformIndexToPhysicalPoint (index, fixed_point);
	moving_point = xf->TransformPoint (fixed_point);
	for (int r = 0; r < Dimension; r++) {
	    displacement[r] = moving_point[r] - fixed_point[r];
	}
	fi.Set (displacement);
	++fi;
    }
    return field;
}

DeformationFieldType::Pointer 
xform_itk_vf_to_itk_vf (DeformationFieldType::Pointer vf,
	         FloatImageType::Pointer image)
{
    printf ("Setting deformation field\n");
    const DeformationFieldType::SpacingType& vf_spacing = vf->GetSpacing();
    printf ("Deformation field spacing is: %g %g %g\n", 
	    vf_spacing[0], vf_spacing[1], vf_spacing[2]);
    const DeformationFieldType::SizeType vf_size = vf->GetLargestPossibleRegion().GetSize();
    printf ("VF Size is %d %d %d\n", vf_size[0], vf_size[1], vf_size[2]);

    const FloatImageType::SizeType& img_size = image->GetLargestPossibleRegion().GetSize();
    printf ("IM Size is %d %d %d\n", img_size[0], img_size[1], img_size[2]);
    const FloatImageType::SpacingType& img_spacing = image->GetSpacing();
    printf ("Deformation field spacing is: %g %g %g\n", 
	    img_spacing[0], img_spacing[1], img_spacing[2]);

    if (vf_size[0] != img_size[0] || vf_size[1] != img_size[1] || vf_size[2] != img_size[2]) {
	//printf ("Deformation stats (pre)\n");
	//deformation_stats (vf);

	vf = vector_resample_image (vf, image);

	//printf ("Deformation stats (post)\n");
	//deformation_stats (vf);
	const DeformationFieldType::SizeType vf_size = vf->GetLargestPossibleRegion().GetSize();
	printf ("NEW VF Size is %d %d %d\n", vf_size[0], vf_size[1], vf_size[2]);
	const FloatImageType::SizeType& img_size = image->GetLargestPossibleRegion().GetSize();
	printf ("IM Size is %d %d %d\n", img_size[0], img_size[1], img_size[2]);
    }
    return vf;
}

/* Here what we're going to do is use GPUIT library to interpolate the 
    B-Spline at its native resolution, then convert gpuit_vf -> itk_vf. */
static DeformationFieldType::Pointer
xform_gpuit_bsp_to_itk_vf (Xform_GPUIT_Bspline *xgb, 
			    FloatImageType::Pointer image)
{
    int i;
    Volume *vf;
    float* img;
    DeformationFieldType::Pointer itk_vf;

    /* GCS FIX: This won't work if roi_offset is not 0 */
    vf = volume_create (xgb->parms.roi_dim, xgb->img_origin, xgb->img_spacing,
			PT_VF_FLOAT_INTERLEAVED, 0);
    bspline_interpolate_vf (vf, &xgb->parms);

    /* Convert from GPUIT_BSP voxel-based to GPUIT_VF mm-based */
    img = (float*) vf->img;
    printf ("Interpolated vfx at 0,0,0 = %g\n", img[0]);
    for (i = 0; i < vf->npix; i++) {
	img[3*i  ] *= xgb->img_spacing[0];
	img[3*i+1] *= xgb->img_spacing[1];
	img[3*i+2] *= xgb->img_spacing[2];
    }

    itk_vf = xform_gpuit_vf_to_itk_vf (vf, image);

    volume_free (vf);
    return itk_vf;
}

/* This is really dumb.  But what we're going to do is 
   to convert gpuit -> itk at the native resolution, 
   then itk -> itk to change resolution.  */
static DeformationFieldType::Pointer 
xform_gpuit_vf_to_itk_vf (Volume* vf,
			    FloatImageType::Pointer image)
{
    int i;
    DeformationFieldType::SizeType sz;
    DeformationFieldType::IndexType st;
    DeformationFieldType::RegionType rg;
    DeformationFieldType::PointType og;
    DeformationFieldType::SpacingType sp;
    DeformationFieldType::Pointer itk_vf = DeformationFieldType::New();
    FloatVectorType displacement;

    /* Copy header & allocate data for itk */
    for (i = 0; i < 3; i++) {
	st[i] = 0;
	sz[i] = vf->dim[i];
	sp[i] = vf->pix_spacing[i];
	og[i] = vf->offset[i];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);

    itk_vf->SetRegions (rg);
    itk_vf->SetOrigin (og);
    itk_vf->SetSpacing (sp);
    itk_vf->Allocate();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (itk_vf, itk_vf->GetBufferedRegion());

    if (vf->pix_type == PT_VF_FLOAT_INTERLEAVED) {
	float* img = (float*) vf->img;
	int i = 0;
	for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi) {
	    for (int r = 0; r < Dimension; r++) {
		displacement[r] = img[i++];
	    }
	    fi.Set (displacement);
	}
    }
    else if (vf->pix_type == PT_VF_FLOAT_PLANAR) {
	float** img = (float**) vf->img;
	int i = 0;
	for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi, ++i) {
	    for (int r = 0; r < Dimension; r++) {
		displacement[r] = img[r][i];
	    }
	    fi.Set (displacement);
	}
    } else {
	print_and_exit ("Irregular pix_type used converting gpuit_xf -> itk\n");
    }

    /* Resample to requested resolution */
    itk_vf = xform_itk_vf_to_itk_vf (itk_vf, image);

    return itk_vf;
}

/* -----------------------------------------------------------------------
   Conversion to gpuit_bsp
   ----------------------------------------------------------------------- */
void
xform_gpuit_bsp_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Xform_GPUIT_Bspline* xgb_new)
{
    Xform xf_tmp;
    Xform_GPUIT_Bspline* xgb_old = xf_in->get_gpuit_bsp();
    BSPLINE_Parms* parms_old = &xgb_old->parms;
    BSPLINE_Data* bspd_old = &parms_old->bspd;
    BSPLINE_Parms* parms_new = &xgb_new->parms;
    BSPLINE_Data* bspd_new = &parms_new->bspd;
    OriginType img_origin;
    SpacingType img_spacing;
    ImageRegionType img_region;

    gpuit_bsp_region_to_itk_bsp_region (img_origin, 
	    img_spacing, img_region, 
	    parms_new->roi_offset, parms_new->roi_dim, 
	    xgb_new->img_origin, xgb_new->img_spacing);

    xform_gpuit_bsp_to_itk_bsp (&xf_tmp, xf_in, img_origin, img_spacing, 
		img_region, xgb_new->grid_spac);

    /* Copy from ITK coefficient array to gpuit coefficient array */
    int k = 0;
    for (int d = 0; d < Dimension; d++) {
	for (int i = 0; i < bspd_new->num_knots; i++) {
	    bspd_new->coeff[3*i+d] = xf_tmp.m_itk_bsp_parms[k] / img_spacing[d];
	    k++;
	}
    }
}


/* -----------------------------------------------------------------------
   Conversion to gpuit_vf
   ----------------------------------------------------------------------- */
Volume*
xform_gpuit_vf_to_gpuit_vf (Volume* vf_in, int* dim, float* offset, float* pix_spacing)
{
    Volume* vf_out;
    vf_out = volume_resample (vf_in, dim, offset, pix_spacing);
    return vf_out;
}


/* -----------------------------------------------------------------------
   Selection routines to convert from X to Y
   ----------------------------------------------------------------------- */
void
xform_to_trn (Xform *xf_out, 
	      Xform *xf_in, 
	      Stage_Parms* stage, 
	      const OriginType& img_origin, 
	      const SpacingType& img_spacing,
	      const ImageRegionType& img_region)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_translation_default (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_TRANSLATION:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_VERSOR:
    case XFORM_ITK_AFFINE:
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to trn\n");
	break;
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit xforms not fully implemented\n");
	break;
    }
}

void
xform_to_vrs (Xform *xf_out, 
	      Xform *xf_in, 
	      Stage_Parms* stage, 
	      const OriginType& img_origin, 
	      const SpacingType& img_spacing,
	      const ImageRegionType& img_region)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_versor_default (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_TRANSLATION:
	print_and_exit ("Sorry, couldn't convert to vrs\n");
	break;
    case XFORM_ITK_VERSOR:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_AFFINE:
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to vrs\n");
	break;
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit xforms not fully implemented\n");
	break;
    }
}

void
xform_to_aff (Xform *xf_out, 
	      Xform *xf_in, 
	      Stage_Parms* stage, 
	      const OriginType& img_origin, 
	      const SpacingType& img_spacing,
	      const ImageRegionType& img_region)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_affine_default (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_trn_to_aff (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_VERSOR:
	xform_vrs_to_aff (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_AFFINE:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to aff\n");
	break;
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit xforms not fully implemented\n");
	break;
    }
}

void
xform_to_itk_bsp (Xform *xf_out, 
		  Xform *xf_in, 
		  Stage_Parms* stage, 
		  const OriginType& img_origin, 
		  const SpacingType& img_spacing,
		  const ImageRegionType& img_region)
{
    BsplineTransformType::Pointer bsp;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_itk_bsp_default (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_trn_to_itk_bsp (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_VERSOR:
	xform_vrs_to_itk_bsp (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_AFFINE:
	xform_aff_to_itk_bsp (xf_out, xf_in, stage, 
				img_origin, img_spacing, img_region);
	break;
    case XFORM_ITK_BSPLINE:
	xform_itk_bsp_to_itk_bsp (xf_out, xf_in, 
				img_origin, img_spacing, img_region, 
				stage->grid_spac);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert itk_vf to itk_bsp\n");
	break;
    case XFORM_GPUIT_BSPLINE:
	xform_gpuit_bsp_to_itk_bsp (xf_out, xf_in, 
				    img_origin, img_spacing, img_region,
				    stage->grid_spac);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert gpuit_vf to itk_bsp\n");
	break;
    }
}

void
xform_to_itk_vf (Xform* xf_out, Xform *xf_in, const int* dim, float* offset, float* spacing)
{
    DeformationFieldType::Pointer vf;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	print_and_exit ("Sorry, couldn't convert to vf\n");
	break;
    case XFORM_ITK_TRANSLATION:
	vf = xform_any_to_itk_vf (xf_in->get_trn(), dim, offset, spacing);
	break;
    case XFORM_ITK_VERSOR:
	vf = xform_any_to_itk_vf (xf_in->get_vrs(), dim, offset, spacing);
	break;
    case XFORM_ITK_AFFINE:
	vf = xform_any_to_itk_vf (xf_in->get_aff(), dim, offset, spacing);
	break;
    case XFORM_ITK_BSPLINE:
	vf = xform_any_to_itk_vf (xf_in->get_bsp(), dim, offset, spacing);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, itk_vf to itk_vf not implemented\n");
	break;
    case XFORM_GPUIT_BSPLINE:
	print_and_exit ("Sorry, gpuit_bsp to itk_vf not implemented\n");
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit_vf to itk_vf not implemented\n");
	break;
    }
    xf_out->set_itk_vf (vf);
}

/* Overloaded fn.  Fix soon... */
void
xform_to_itk_vf (Xform* xf_out, Xform *xf_in, FloatImageType::Pointer image)
{
    DeformationFieldType::Pointer vf;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	print_and_exit ("Sorry, couldn't convert to vf\n");
	break;
    case XFORM_ITK_TRANSLATION:
	vf = xform_any_to_itk_vf (xf_in->get_trn(), image);
	break;
    case XFORM_ITK_VERSOR:
	vf = xform_any_to_itk_vf (xf_in->get_vrs(), image);
	break;
    case XFORM_ITK_AFFINE:
	vf = xform_any_to_itk_vf (xf_in->get_aff(), image);
	break;
    case XFORM_ITK_BSPLINE:
	vf = xform_any_to_itk_vf (xf_in->get_bsp(), image);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	vf = xform_itk_vf_to_itk_vf (xf_in->get_itk_vf(), image);
	break;
    case XFORM_GPUIT_BSPLINE:
	vf = xform_gpuit_bsp_to_itk_vf (xf_in->get_gpuit_bsp(), image);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	vf = xform_gpuit_vf_to_itk_vf (xf_in->get_gpuit_vf(), image);
	break;
    }
    xf_out->set_itk_vf (vf);
}

void
xform_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Xform_GPUIT_Bspline* xgb_new)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	/* Do nothing */
	break;
    case XFORM_ITK_TRANSLATION:
	print_and_exit ("Sorry, itk_translation to gpuit_bsp not implemented\n");
	break;
    case XFORM_ITK_VERSOR:
	print_and_exit ("Sorry, itk_versor to gpuit_bsp not implemented\n");
	break;
    case XFORM_ITK_AFFINE:
	print_and_exit ("Sorry, itk_affine to gpuit_bsp not implemented\n");
	break;
    case XFORM_ITK_BSPLINE:
	print_and_exit ("Sorry, itk_bsp to gpuit_bsp not implemented\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, itk_vf to gpuit_bsp not implemented\n");
	break;
    case XFORM_GPUIT_BSPLINE:
	xform_gpuit_bsp_to_gpuit_bsp (xf_out, xf_in, xgb_new);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit_vf to gpuit_bsp not implemented\n");
	break;
    }
    xf_out->set_gpuit_bsp (xgb_new);
}

void
xform_to_gpuit_vf (Xform* xf_out, Xform *xf_in, int* dim, float* offset, float* pix_spacing)
{
    Volume* vf = 0;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	print_and_exit ("Sorry, couldn't convert NONE to gpuit_vf\n");
	break;
    case XFORM_ITK_TRANSLATION:
	print_and_exit ("Sorry, itk_translation to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_VERSOR:
	print_and_exit ("Sorry, itk_versor to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_AFFINE:
	print_and_exit ("Sorry, itk_affine to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_BSPLINE:
	print_and_exit ("Sorry, itk_bspline to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, itk_vf to gpuit_vf not implemented\n");
	break;
    case XFORM_GPUIT_BSPLINE:
	print_and_exit ("Sorry, gpuit_bspline to gpuit_vf not implemented\n");
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	vf = xform_gpuit_vf_to_gpuit_vf (xf_in->get_gpuit_vf(), dim, offset, pix_spacing);
	break;
    }

    xf_out->set_gpuit_vf (vf);
}
