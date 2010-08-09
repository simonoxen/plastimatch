/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This is an alternate version of merge_affine_vector_field (gcs, 3/3/05) */
#include "plm_config.h"
#include <fstream>
#include <string>

#include "itkImageFileReader.h" 
#include "itkImageFileWriter.h" 
#include "itkAffineTransform.h"
#include "itkImageRegionIterator.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineDeformableTransform.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"

const unsigned int Dimension = 3;

int main( int argc, char *argv[] )
{
    if( argc < 4 )
    {
	std::cerr << "Missing Parameters " << std::endl;
	std::cerr << "Usage: " << argv[0];
	std::cerr << " affine_xform.txt vector_field.mha";
	std::cerr << " output_vector_field.mha " << std::endl;
	return 1;
    }

    //load affine parameter file, build affine transform
    std::ifstream paramFile(argv[1]) ;
    if ( !paramFile )
    {
	std::cout << "ERROR: cannot open the parameter file" << std::endl ;
	return 0;
    } 

    typedef itk::AffineTransform<float,3> Affine3DType;
    Affine3DType::Pointer aff3 = Affine3DType::New();

    Affine3DType::ParametersType parameters( aff3->GetNumberOfParameters() );

    //get header line first
    std::string header ;
    std::getline(paramFile, header) ;
    int i = 0;
    while ( !paramFile.eof() ) 
    {    
	paramFile >> parameters[i];
	printf("para[%d] = %f\n", i, parameters[i] );
	++i ;
    } // end of while

    aff3->SetParameters( parameters );

    //load vector field file

    typedef itk::Vector< float, Dimension >  VectorType;
    typedef itk::Image< VectorType, Dimension >  DeformationFieldType;
    typedef itk::ImageFileReader< DeformationFieldType >  FieldReaderType;

    FieldReaderType::Pointer fieldReader = FieldReaderType::New();
    fieldReader->SetFileName( argv[2] );

    try 
    {
	fieldReader->Update();
    }
    catch (itk::ExceptionObject& excp) 
    {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
	return 0;
    }
    DeformationFieldType::Pointer deform_field = fieldReader->GetOutput();

    DeformationFieldType::Pointer field = DeformationFieldType::New();
    field->SetRegions (deform_field->GetBufferedRegion());
    field->SetOrigin (deform_field->GetOrigin());
    field->SetSpacing (deform_field->GetSpacing());
    field->Allocate();

    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (field, deform_field->GetBufferedRegion());
    FieldIterator d_fi (deform_field, deform_field->GetBufferedRegion());

    fi.GoToBegin();
    d_fi.GoToBegin();

    const unsigned int SplineDimension = Dimension;
    const unsigned int SplineOrder = 3;
    typedef float CoordinateRepType;
    typedef itk::BSplineDeformableTransform<
	    CoordinateRepType,
	    SplineDimension,
	    SplineOrder > BsplineTransformType;

    BsplineTransformType::InputPointType fixed_point;
    BsplineTransformType::OutputPointType moving_point_1;
    BsplineTransformType::OutputPointType moving_point_2;
    BsplineTransformType::OutputPointType m_point;
  

    DeformationFieldType::IndexType index;

    VectorType displacement;

#if defined (commentout)
    typedef itk::NearestNeighborInterpolateImageFunction <
	DeformationFieldType, double > InterpolatorType;
    typedef itk::VectorNearestNeighborInterpolateImageFunction <
	DeformationFieldType, double > InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
#endif
    typedef itk::VectorLinearInterpolateImageFunction < 
	DeformationFieldType, float > VectorInterpolatorType;
    VectorInterpolatorType::Pointer interpolator = VectorInterpolatorType::New();
    interpolator->SetInputImage (deform_field);
    VectorInterpolatorType::OutputType interp_disp;

    while (!d_fi.IsAtEnd()) 
    {
	index = d_fi.GetIndex();
	deform_field->TransformIndexToPhysicalPoint (index, fixed_point);

	moving_point_1 = aff3->TransformPoint (fixed_point);

	interp_disp = interpolator->Evaluate(moving_point_1);

	for (int r = 0; r < 3; r++) 
	{
	    displacement[r] = moving_point_1[r] + interp_disp[r] - fixed_point[r];
	}

	fi.Set (displacement);
	++fi;
	++d_fi;
    }

    typedef itk::ImageFileWriter< DeformationFieldType >  FieldWriterType;
    FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
    fieldWriter->SetInput (field);
    fieldWriter->SetFileName (argv[3]);
    try 
    {
	fieldWriter->Update();
    }
    catch (itk::ExceptionObject& excp) 
    {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
    }
    return 0;
}

