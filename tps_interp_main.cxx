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



void print_usage (void)
{
	/*printf ("This executable computes the DICE coefficient for 2 give binary volumes \n");*/
	printf ("Usage: tps_interp\n");
	printf ("  original landmarks\t");
	printf ("  target landmarks\t");
	printf ("  OriginalImage\t");
	printf ("  WarpedImage\t");
	printf ("  VectorField\t");
	exit (-1);
}


void tps_interp_main(TPS_parms* parms){

	//FILE* reference;
	//FILE* target;
	
	//reference=fopen(parms->reference,"r");
	//target=fopen(parms->target,"r");

	////printf("ref: %s\n",argv[1]);
	////printf("target: %s\n", argv[2]);

	////system("PAUSE");
	//if(!reference || !target){
	//	fprintf(stderr, "An error occurred while opening the landmark files!");
	//	exit(-1);
	//}
	DeformationFieldType::Pointer vf = DeformationFieldType::New();

    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    itk__GetImageType (parms->original, pixelType, componentType);
    switch (componentType) {
	case itk::ImageIOBase::UCHAR:
	    {
		UCharImageType::Pointer img_in = load_uchar (parms->original);
		//unsigned char foo;
		do_tps(parms,img_in,(unsigned char)0);
	    }
	    break;
        case itk::ImageIOBase::SHORT:
	    {
		ShortImageType::Pointer img_in = load_short (parms->original);
		//short foo;
		do_tps(parms,img_in,(short)-1200);
	    }
	    break;
        case itk::ImageIOBase::FLOAT:
	    {
		FloatImageType::Pointer img_in = load_float (parms->original);
		//float foo;
		do_tps(parms,img_in,(float)-1200.0);
	    }
	    break;
	default:
	    printf ("Error, unsupported output type\n");
	    exit (-1);
	    break;
    }

}

int main(int argc, char* argv[]){

	//FILE* reference;
	//FILE* target;

	if(argc<6){
		print_usage();
		exit(-1);
	}else{
		TPS_parms* parms=(TPS_parms*)malloc(sizeof(TPS_parms));
		memset(parms,0,sizeof(TPS_parms));
		//parms->reference=argv[1];
		//parms->target=argv[2];
		parms->reference=argv[1];
		parms->target=argv[2];
		parms->original=argv[3];
		parms->warped=argv[4];
		parms->vf=argv[5];
		tps_interp_main(parms);
	}

	//parms->reference=fopen(argv[1],"r");
	//parms->target=fopen(argv[2],"r");

	//printf("ref: %s\n",argv[1]);
	//printf("target: %s\n", argv[2]);

	////system("PAUSE");
	//if(!reference || !target){
	//	fprintf(stderr, "An error occurred while opening the landmark files!");
	//	exit(-1);
	//}

	//FloatImageType::Pointer img_in = FloatImageType::New();
	//reader->SetFileName( argv[3] );
	//do_tps(reference,target);
	

	return 0;
}
