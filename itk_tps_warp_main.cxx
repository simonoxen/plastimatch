/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itk_tps.h"

void print_usage (void)
{
    printf ("Usage: tps_warp\n");
    printf ("  reference landmarks ");
    printf ("  target landmarks ");
    printf ("  fixed image ");
    printf ("  movingImage ");
    printf ("  output Image ");
    printf ("  output VectorField\n");
    exit (-1);
}


void 
tps_interp_main (TPS_parms* parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();

    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    itk__GetImageType (parms->moving, pixelType, componentType);
    switch (componentType) {
    case itk::ImageIOBase::UCHAR:
	{
	    UCharImageType::Pointer img_moving = load_uchar (parms->moving, 0);
	    UCharImageType::Pointer img_fixed = load_uchar (parms->fixed, 0);
	    do_tps(parms,img_fixed,img_moving,(unsigned char)0);
	}
	break;
    case itk::ImageIOBase::SHORT:
	{
	    ShortImageType::Pointer img_moving = load_short (parms->moving, 0);
	    ShortImageType::Pointer img_fixed = load_short (parms->fixed, 0);
	    do_tps(parms,img_fixed,img_moving,(short)-1200);
	}
	break;
    case itk::ImageIOBase::FLOAT:
	{
	    FloatImageType::Pointer img_moving = load_float (parms->moving, 0);
	    FloatImageType::Pointer img_fixed = load_float (parms->fixed, 0);
	    do_tps(parms,img_fixed,img_moving,(float)-1200.0);
	}
	break;
    default:
	printf ("Error, unsupported output type\n");
	exit (-1);
	break;
    }
}

int 
main (int argc, char* argv[])
{
    if (argc < 6) {
	print_usage ();
	exit (-1);
    }

    TPS_parms* parms=(TPS_parms*)malloc(sizeof(TPS_parms));
    memset(parms,0,sizeof(TPS_parms));
    parms->reference=argv[1];
    parms->target=argv[2];
    parms->fixed=argv[3];
    parms->moving=argv[4];
    parms->warped=argv[5];
    parms->vf=argv[6];
    tps_interp_main(parms);
    free (parms);

    return 0;
}
