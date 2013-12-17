/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_jacobian_h_
#define _vf_jacobian_h_

#include "plmbase_config.h"
#include "itk_image_type.h"
#include "pstring.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
// #include <itkVectorResampleImageFilter.h>
// #include <itkAffineTransform.h>
// #include <itkVectorNearestNeighborInterpolateImageFunction.h>
// #include <itkLinearInterpolateImageFunction.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>

class Plm_image;
typedef itk::DisplacementFieldJacobianDeterminantFilter<DeformationFieldType, float> JacobianFilterType;

class PLMBASE_API Jacobian_stats {
public:
    float min;
    float max;
    Pstring outputstats_fn;
public:
    Jacobian_stats () {
	outputstats_fn = " ";
	min=0;
	max=0;
    }
};

class PLMBASE_API Jacobian {
public:
    /*Xform * */
    DeformationFieldType::Pointer vf;
    Pstring vfjacstats_fn;
    float jacobian_min;
    float jacobian_max;

public:
    Jacobian();
    void set_input_vf (DeformationFieldType::Pointer vf);
    void set_output_vfstats_name (Pstring vfjacstats);
    FloatImageType::Pointer make_jacobian ();
private:  
    void write_output_statistics(Jacobian_stats *);
};

#endif
