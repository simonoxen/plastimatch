/*===========================================================
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
===========================================================*/

#include "tps_interp.h"

void print_usage (void)
{
	printf ("Usage: tps_interp\n");
	printf ("  original landmarks\t");
	printf ("  target landmarks\t");
	printf ("  OriginalImage\t");
	printf ("  WarpedImage\t");
	printf ("  VectorField\t");
	exit (-1);
}


void tps_interp_main(TPS_parms* parms){

	DeformationFieldType::Pointer vf = DeformationFieldType::New();

    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    itk__GetImageType (parms->original, pixelType, componentType);
    switch (componentType) {
	case itk::ImageIOBase::UCHAR:
	    {
		UCharImageType::Pointer img_in = load_uchar (parms->original);
		do_tps(parms,img_in,(unsigned char)0);
	    }
	    break;
        case itk::ImageIOBase::SHORT:
	    {
		ShortImageType::Pointer img_in = load_short (parms->original);
		do_tps(parms,img_in,(short)-1200);
	    }
	    break;
        case itk::ImageIOBase::FLOAT:
	    {
		FloatImageType::Pointer img_in = load_float (parms->original);
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

	if(argc<6){
		print_usage();
		exit(-1);
	}else{
		TPS_parms* parms=(TPS_parms*)malloc(sizeof(TPS_parms));
		memset(parms,0,sizeof(TPS_parms));
		parms->reference=argv[1];
		parms->target=argv[2];
		parms->original=argv[3];
		parms->warped=argv[4];
		parms->vf=argv[5];
		tps_interp_main(parms);
	}

	return 0;
}
