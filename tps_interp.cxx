#include "tps_interp.h"

//typedef   itk::Vector< float, 3 >  FieldVectorType;
//typedef   itk::Image< FieldVectorType,  3 >   DeformationFieldType;
//typedef   itk::ImageFileWriter< DeformationFieldType >  FieldWriterType;
//typedef double CoordinateRepType;
//typedef itk::ThinPlateSplineKernelTransform<CoordinateRepType,3> TransformType;
//typedef itk::Point<CoordinateRepType,3 > PointType;
//typedef std::vector<PointType > PointArrayType;
//typedef TransformType::PointSetType PointSetType;
//typedef PointSetType::Pointer PointSetPointer;
//typedef PointSetType::PointIdentifier PointIdType;
//typedef itk::ResampleImageFilter<InputImageType,InputImageType> ResamplerType;
//typedef itk::LinearInterpolateImageFunction<ImgType, float >  InterpolatorType;
//typedef itk::LinearInterpolateImageFunction<FloatImageType, float >  FloatInterpolatorType;
//typedef itk::LinearInterpolateImageFunction<UCharImageType, unsigned char >  UCharInterpolatorType;
//typedef itk::LinearInterpolateImageFunction<ShortImageType, short >  ShortInterpolatorType;

template<class T, class U>
U do_tps(TPS_parms* parms,T img_in,U){

	typedef double CoordinateRepType;
	typedef typename itk::ThinPlateSplineKernelTransform<CoordinateRepType,3> TransformType;
	typedef typename itk::Point<CoordinateRepType,3 > PointType;
	typedef typename std::vector<PointType > PointArrayType;
	typedef typename TransformType::PointSetType PointSetType;
	typedef typename PointSetType::Pointer PointSetPointer;
	typedef typename PointSetType::PointIdentifier PointIdType;


	PointType p1;  
	int dim[3];
	float offset[3];
	float spacing[3];
	char line[BUFLEN];
	Xform xform_tmp, xform;

	get_image_header(dim, offset, spacing, img_in);

	// Define container for landmarks
	PointSetType::Pointer sourceLandMarks = PointSetType::New();
	PointSetType::Pointer targetLandMarks = PointSetType::New();

	PointSetType::PointsContainer::Pointer sourceLandMarkContainer = sourceLandMarks->GetPoints();
	PointSetType::PointsContainer::Pointer targetLandMarkContainer = targetLandMarks->GetPoints();

	PointIdType id = itk::NumericTraits< PointIdType >::Zero;

	while(fgets(line, BUFLEN,parms->reference)){
		if(sscanf(line,"%f %f %f",&p1[0],&p1[1],&p1[2])==3){
			sourceLandMarkContainer->InsertElement( id++, p1 );
			printf("reference Landmark: %f %f %f \n",p1[0],p1[1],p1[2]);
		}else{
			printf("PUNTO: %f %f %f \n",&p1[0],&p1[1],&p1[2]);
			printf("Error! can't read the reference landmarks file");
			exit(-1);
		}
	}
	id = itk::NumericTraits< PointIdType >::Zero;
	while(fgets(line, BUFLEN,parms->target)){
		if(sscanf(line,"%f %f %f",&p1[0],&p1[1],&p1[2])==3){
			targetLandMarkContainer->InsertElement( id++, p1 );
			printf("target Landmark: %f %f %f \n",p1[0],p1[1],p1[2]);
		}else{
			printf("Error! can't read the target landmarks file");
			exit(-1);
		}
	}

	fclose(parms->reference);
	fclose(parms->target);

	
	TransformType::Pointer tps = TransformType::New();
	tps->SetSourceLandmarks(sourceLandMarks);
	tps->SetTargetLandmarks(targetLandMarks);
	tps->ComputeWMatrix();

	xform.set_itk_tps(tps);
	xform_to_itk_vf (&xform_tmp,&xform, dim, offset, spacing);

	DeformationFieldType::Pointer vf = DeformationFieldType::New();
	vf = xform_tmp.get_itk_vf();

	printf ("Warping...\n");
	//InterpolatorType::Pointer interpolator = InterpolatorType::New();
	
	T im_warped=itk_warp_image (img_in, vf, 1, -1200);

	printf ("Saving...\n");
    save_image (im_warped, parms->warped);
    save_image(vf, parms->vf);

	return U;
	}

/* Explicit instantiations */
/* RMK: Visual studio 2005 without service pack requires <float> specifier
   on the explicit extantiations.  The current hypothesis is that this 
   is because the template is nested. */
template unsigned char do_tps<unsigned char>(TPS_parms* parms,UCharImageType::Pointer img_in, unsigned char);
template float do_tps<float>(TPS_parms* parms,FloatImageType::Pointer img_in, float);
template short do_tps<short>(TPS_parms* parms,ShortImageType::Pointer img_in, short);