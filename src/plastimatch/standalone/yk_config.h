#ifndef YK_CONFIG_H
#define YK_CONFIG_H


//#include "itkImage.h"
//#include "itkImageFileReader.h"
//#include "itkImageFileWriter.h"



#define DEFAULT_LABEL_SIZE1 512
#define DEFAULT_LABEL_SIZE2 256
#define DEFAULT_LABEL_SIZE3 256

#define MAX_LINE_LENGTH 1024


#define GY2YKIMG_MAG 700.0 //if 1000: 70Gy --> 70000 --> overflow,  if 500: 70Gy --> 35000 cGy, if 700: 100Gy --> 70000 cGy ->OK
#define GAMMA2YKIMG_MAG 1000.0 //2 --> 512: 0~1: 256, 1-2: 256
#define GY2CGY 100.0;

#define NON_NEG_SHIFT 0.0 //2 --> 512: 0~1: 256, 1-2: 256

#define DEFAULT_LABEL_WIDTH 256
#define DEFAULT_LABEL_HEIGHT 256

struct VEC3D{
    double x;
    double y;
    double z;
};


enum enPLANE{
    PLANE_AXIAL = 0,
    PLANE_FRONTAL,
    PLANE_SAGITTAL,
};


enum enPROFILE_DIRECTON{
    PRIFLE_HOR = 0,
    PRIFLE_VER,
};


enum enViewArrange{
    AXIAL_FRONTAL_SAGITTAL = 0,
    FRONTAL_SAGITTAL_AXIAL,
    SAGITTAL_AXIAL_FRONTAL,
};

enum enRegisterOption{
    PLAST_RIGID = 0,
    PLAST_GRADIENT,
    PLAST_AFFINE,
    PLAST_BSPLINE,
};


#endif // YK_CONFIG_H



