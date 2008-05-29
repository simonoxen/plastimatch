/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This is the ITK implementation.  See these:

    http://public.kitware.com/pipermail/insight-users/2005-June/013527.html
    http://public.kitware.com/pipermail/insight-users/2005-October/015346.html
    http://public.kitware.com/pipermail/insight-users/2006-February/016592.html

    Regarding the use of char* buffers, see the following:

    http://www.itk.org/CourseWare/Training/GettingStarted-V.pdf
    http://public.kitware.com/pipermail/insight-users/2002-December/001890.html

    The itk implementation (2.4.1) crashes on my data.  See test case below.
  */
#include "itkImage.h"
#include "itkPolyLineParametricPath.h"
#include "itkPolylineMask2DImageFilter.h"

typedef itk::Image < unsigned char, 2 > ImageType;
typedef itk::PolyLineParametricPath<2> PolylineType;
typedef PolylineType::VertexType VertexType;
typedef itk::PolylineMask2DImageFilter < ImageType, PolylineType,
                           ImageType > MaskFilterType;

void
render_slice_polyline (unsigned char* acc_img,
		    int* dims,
		    float* spacing,
		    float* offset, 
		    int num_vertices,
		    float* x,
		    float* y)
{
#if defined (commentout)
    int i;

    /* Make an input image */
    ImageType::Pointer img1 = ImageType::New();
    ImageType::IndexType start = { 0, 0 };
    ImageType::SizeType size = { dims[0], dims[1] };
    ImageType::RegionType region;
    region.SetSize (size);
    region.SetIndex (start);
//    img1->SetRegions (region);
    img1->SetLargestPossibleRegion (region);
    img1->SetBufferedRegion (region);
    img1->SetRequestedRegion (region);
  
    ImageType::PointType origin;
    origin[0] = offset[0];
    origin[1] = offset[1];
    img1->SetOrigin (origin);

    printf ("Setting spacing: %g %g\n", spacing[0], spacing[1]);
    ImageType::SpacingType itkspacing;
    itkspacing[0] = spacing[0];
    itkspacing[1] = spacing[1];
    img1->SetSpacing (spacing);

    img1->Allocate ();
    img1->FillBuffer (1);

    /* Make a polyline.  Apparently AddVertex() is the only way to do this.  */
    PolylineType::Pointer polyline = PolylineType::New();
    for (i = 0; i < num_vertices; i++) {
	printf ("Adding vertex (%d)\n", i);
	VertexType v;
	v[0] = x[i];
	v[1] = y[i];
	polyline->AddVertex(v);
    }

    // Create a mask  Filter
    MaskFilterType::Pointer filter = MaskFilterType::New();

    // Connect the input image
    filter->SetInput1 (img1);

    // Connect the Polyline
    filter->SetInput2 (polyline);
    filter->Update();
#endif


 // Declare the types of the images
  typedef itk::Image<unsigned char, 2>     inputImageType;
  typedef itk::Image<unsigned char, 2>     outputImageType;
  typedef itk::PolyLineParametricPath<2>     inputPolylineType;

  // Declare the type of the index to access images
  typedef inputImageType::IndexType         inputIndexType;

  // Declare the type of the size 
  typedef inputImageType::SizeType          inputSizeType;

  // Declare the type of the Region
  typedef inputImageType::RegionType         inputRegionType;

  // Create images
  inputImageType::Pointer inputImage    = inputImageType::New();
 
  // Create polyline
  inputPolylineType::Pointer inputPolyline   = inputPolylineType::New();

  // Define their size, and start index
  inputSizeType size;
  size[0] = 512;
  size[1] = 512;


  inputIndexType start;
  start[0] = 0;
  start[1] = 0;


  inputRegionType region;
  region.SetIndex( start );
  region.SetSize( size );

  // Initialize input image
  inputImage->SetLargestPossibleRegion( region );
  inputImage->SetBufferedRegion( region );
  inputImage->SetRequestedRegion( region );
  inputImage->Allocate();
  inputImage->FillBuffer(0);

  // Declare Iterator types apropriated for each image 
  typedef itk::ImageRegionIteratorWithIndex<inputImageType>  inputIteratorType;

  // Create one iterator for Image A (this is a light object)
  inputIteratorType it( inputImage, inputImage->GetBufferedRegion() );
  it.GoToBegin();
  while( !it.IsAtEnd() ) 
    {
    /* fill in only the upper part of the image */
      if(it.GetIndex()[1] > 256)
        {
        it.Set( 255 );
        }
      ++it;
    }

  // Initialize the polyline 
  typedef inputPolylineType::VertexType VertexType;
    
  // Add vertices to the polyline
  VertexType v;
//  v[0] = 128;
  v[0] = -1;
  v[1] = 256;
  inputPolyline->AddVertex(v);
  
  v[0] = 256;
  v[1] = 394;
  inputPolyline->AddVertex(v);
  
  v[0] = 394;
  v[1] = 256;
  inputPolyline->AddVertex(v);

  v[0] = 256;
  v[1] = 128;
  inputPolyline->AddVertex(v);
  
  // Declare the type for the Mask image filter
  typedef itk::PolylineMask2DImageFilter<
                           inputImageType, inputPolylineType,   
                           outputImageType  >     inputFilterType;
            

  // Create a mask  Filter                                
  inputFilterType::Pointer filter = inputFilterType::New();

  // Connect the input image
  filter->SetInput1    ( inputImage ); 
 
  // Connect the Polyline 
  filter->SetInput2    ( inputPolyline ); 
  filter->Update();



}
