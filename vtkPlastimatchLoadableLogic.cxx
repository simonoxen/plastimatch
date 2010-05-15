/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkExampleLoadableModuleLogic.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>
#include "vtkObjectFactory.h"
#include "vtkPlastimatchLoadableLogic.h"
#include "vtkITKGradientAnisotropicDiffusionImageFilter.h"
#include "vtkPlastimatchLoadable.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "plm_register_loadable.h"

vtkPlastimatchLoadableLogic* vtkPlastimatchLoadableLogic::New()
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance (
	"vtkPlastimatchLoadableLogic");
    if (ret) {
	return (vtkPlastimatchLoadableLogic*)ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkPlastimatchLoadableLogic;
}


//----------------------------------------------------------------------------
vtkPlastimatchLoadableLogic::vtkPlastimatchLoadableLogic()
{
    this->PlastimatchLoadableNode = NULL;
}

//----------------------------------------------------------------------------
vtkPlastimatchLoadableLogic::~vtkPlastimatchLoadableLogic()
{
    vtkSetMRMLNodeMacro(this->PlastimatchLoadableNode, NULL);
}

//----------------------------------------------------------------------------
void vtkPlastimatchLoadableLogic::PrintSelf(ostream& os, vtkIndent indent)
{
}

void vtkPlastimatchLoadableLogic::Apply()
{
    // check if MRML node is present 
    if (this->PlastimatchLoadableNode == NULL)
    {
	vtkErrorMacro("No input PlastimatchLoadableNode found");
	return;
    }
  
    // find fixed volume
    vtkMRMLScalarVolumeNode *fixed_volume 
	= vtkMRMLScalarVolumeNode::SafeDownCast (
	    this->GetMRMLScene()->GetNodeByID (
		this->PlastimatchLoadableNode->GetFixedVolumeRef()));
    if (fixed_volume == NULL)
    {
	vtkErrorMacro("No fixed volume found");
	return;
    }
  
    // find moving volume
    vtkMRMLScalarVolumeNode *moving_volume 
	= vtkMRMLScalarVolumeNode::SafeDownCast (
	    this->GetMRMLScene()->GetNodeByID (
		this->PlastimatchLoadableNode->GetMovingVolumeRef()));
    if (moving_volume == NULL)
    {
	vtkErrorMacro("No moving volume found");
	return;
    }
  
    // find output volume
    vtkMRMLScalarVolumeNode *outVolume 
	=  vtkMRMLScalarVolumeNode::SafeDownCast (
	    this->GetMRMLScene()->GetNodeByID (
		this->PlastimatchLoadableNode->GetOutputVolumeRef()));
    if (outVolume == NULL)
    {
	vtkErrorMacro("No output volume found with id= " 
	    << this->PlastimatchLoadableNode->GetOutputVolumeRef());
	return;
    }

    // copy RASToIJK matrix, and other attributes from input to output
    std::string name (outVolume->GetName());
    std::string id (outVolume->GetID());

#if defined (commentout)
    plm_register_loadable (fixed_volume->GetImageData(),
	moving_volume->GetImageData());
#endif
    plm_register_loadable ();

    outVolume->CopyOrientation (fixed_volume);
    outVolume->SetAndObserveTransformNodeID (
	fixed_volume->GetTransformNodeID());
    outVolume->SetName(name.c_str());
    //outVolume->SetID(id.c_str());

#if defined (commentout)
    // create filter
    //vtkITKGradientAnisotropicDiffusionImageFilter* filter = vtkITKGradientAnisotropicDiffusionImageFilter::New();
    this->GradientAnisotropicDiffusionImageFilter = vtkITKGradientAnisotropicDiffusionImageFilter::New();

    // set filter input and parameters
    this->GradientAnisotropicDiffusionImageFilter->SetInput(inVolume->GetImageData());

    this->GradientAnisotropicDiffusionImageFilter->SetConductanceParameter(this->PlastimatchLoadableNode->GetConductance());
    this->GradientAnisotropicDiffusionImageFilter->SetNumberOfIterations(this->PlastimatchLoadableNode->GetNumberOfIterations());
    this->GradientAnisotropicDiffusionImageFilter->SetTimeStep(this->PlastimatchLoadableNode->GetTimeStep()); 

    // run the filter
    this->GradientAnisotropicDiffusionImageFilter->Update();

    // set ouput of the filter to VolumeNode's ImageData
    // TODO FIX the bug of the image is deallocated unless we do DeepCopy
    vtkImageData* image = vtkImageData::New(); 
    image->DeepCopy( this->GradientAnisotropicDiffusionImageFilter->GetOutput() );
    outVolume->SetAndObserveImageData(image);
    image->Delete();
    outVolume->SetModifiedSinceRead(1);

    //outVolume->SetImageData(this->GradientAnisotropicDiffusionImageFilter->GetOutput());

    // delete the filter
    this->GradientAnisotropicDiffusionImageFilter->Delete();
#endif
}
