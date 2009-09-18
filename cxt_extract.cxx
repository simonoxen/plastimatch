/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkContourExtractor2DImageFilter.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "slice_extract.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "cxt_io.h"
#include "cxt_extract.h"

template<class T>
void
cxt_extract (Cxt_structure_list *structures, T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ContourExtractor2DImageFilter<ULongImage2DType> 
	    ContourType;
    typedef ContourType::VertexType VertexType;
    typedef itk::ImageSliceConstIteratorWithIndex<ImageType> IteratorType;

    IteratorType itSlice (image, image->GetLargestPossibleRegion());
    itSlice.SetFirstDirection(0);
    itSlice.SetSecondDirection(1);
	
    while (!itSlice.IsAtEnd())
    {
	typename ImageType::IndexType k;
	k = itSlice.GetIndex();
	printf("%2d\n", k[2]);
		
	ULongImage2DType::Pointer slice;
	slice = slice_extract (image, k[2], (unsigned long) 0);

	ContourType::Pointer contour = ContourType::New();

	contour->SetContourValue(0.5);
	contour->SetInput(slice);
		
	try
	{
	    contour->Update();
	    //std::cout << "Cerco il contorno!\n" << std::endl;
	}
	catch (itk::ExceptionObject &err)
	{
	    std::cout << "ExceptionObject caught !" << std::endl; 
	    std::cout << err << std::endl; 
	    return;
	}

#if defined (commentout)		
	//std::cout << "NR OUTPUTS:"<<contour->GetNumberOfOutputs() << std::endl; 
	//system("PAUSE");
	for(unsigned int i = 0; i < contour->GetNumberOfOutputs(); i++)
	{
	    ContourType::VertexListConstPointer vertices =contour->GetOutput(i)->GetVertexList();
	    /*fprintf(fp,"%s %d%s%d\n","Contour",k[2],".",i);*/
	    /*fprintf(fp,"%d%s%d\n",k[2],".",i);*/
	    //fprintf(fp,"\n");
	    fprintf(fp,"%s %s %s\n","NaN","NaN","NaN");
	    fprintf(file,"%s %s %s\n","NaN","NaN","NaN");
	    for(unsigned int j = 0; j < vertices->Size(); j++)
	    {
		const VertexType& vertex = vertices->ElementAt(j);
					
		fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]);
		fprintf(file,"%.3f %.3f %.3f \n",vertex[0]*image->GetSpacing()[0]+image->GetSpacing()[3],vertex[1]*image->GetSpacing()[1]+image->GetSpacing()[4],k[2]*image->GetSpacing()[2]+image->GetSpacing()[5]);
		
		//fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]*image->GetSpacing()[2]);
		//std::cout << vertex[0] <<" "<<vertex[1]<<" "<<k[2]<<std::endl;
					

	    }
	    //system("PAUSE");
	}
#endif
	itSlice.NextSlice();
    }
}

/* Explicit instantiations */
template plastimatch1_EXPORT void cxt_extract (Cxt_structure_list *structures, ULongImageType::Pointer image);
