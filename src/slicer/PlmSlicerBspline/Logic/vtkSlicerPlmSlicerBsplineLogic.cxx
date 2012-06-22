/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
// ModuleTemplate includes
#include "vtkSlicerPlmSlicerBsplineLogic.h"
#include "vtkMRMLPlmSlicerBsplineParametersNode.h"

// MRML includes
#include <vtkMRMLVolumeNode.h>

// VTK includes
#include <vtkNew.h>

// STD includes
#include <cassert>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerPlmSlicerBsplineLogic);

//----------------------------------------------------------------------------
vtkSlicerPlmSlicerBsplineLogic::vtkSlicerPlmSlicerBsplineLogic()
{
}

//----------------------------------------------------------------------------
vtkSlicerPlmSlicerBsplineLogic::~vtkSlicerPlmSlicerBsplineLogic()
{
}

//----------------------------------------------------------------------------
void vtkSlicerPlmSlicerBsplineLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//---------------------------------------------------------------------------
void vtkSlicerPlmSlicerBsplineLogic::InitializeEventListeners()
{
  vtkNew<vtkIntArray> events;
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  events->InsertNextValue(vtkMRMLScene::EndBatchProcessEvent);
  this->SetAndObserveMRMLSceneEventsInternal(this->GetMRMLScene(), events.GetPointer());
}

//-----------------------------------------------------------------------------
void vtkSlicerPlmSlicerBsplineLogic::RegisterNodes()
{
  assert(this->GetMRMLScene() != 0);
}

//---------------------------------------------------------------------------
void vtkSlicerPlmSlicerBsplineLogic::UpdateFromMRMLScene()
{
  assert(this->GetMRMLScene() != 0);
}

//---------------------------------------------------------------------------
void vtkSlicerPlmSlicerBsplineLogic
::OnMRMLSceneNodeAdded(vtkMRMLNode* vtkNotUsed(node))
{
}

//---------------------------------------------------------------------------
void vtkSlicerPlmSlicerBsplineLogic
::OnMRMLSceneNodeRemoved(vtkMRMLNode* vtkNotUsed(node))
{
}

//----------------------------------------------------------------------------
//int vtkSlicerPlmSlicerBsplineLogic::Apply()
int vtkSlicerPlmSlicerBsplineLogic::Apply(vtkMRMLPlmSlicerBsplineParametersNode* pnode)
{
  vtkMRMLScene *scene = this->GetMRMLScene();

  fprintf (stderr, ">>> %s\n", pnode->GetFixedVolumeNodeID());

#if 0
  vtkMRMLVolumeNode *fixedInputVolume = 
    vtkMRMLVolumeNode::SafeDownCast(scene->GetNodeByID(pnode->GetFixedVolumeNodeID()));
  vtkMRMLVolumeNode *movingInputVolume = 
    vtkMRMLVolumeNode::SafeDownCast(scene->GetNodeByID(pnode->GetMovingVolumeNodeID()));
  vtkMRMLVolumeNode *warpedInputVolume = 
    vtkMRMLVolumeNode::SafeDownCast(scene->GetNodeByID(pnode->GetWarpedVolumeNodeID()));
  vtkMRMLVolumeNode *xformInputVolume = 
    vtkMRMLVolumeNode::SafeDownCast(scene->GetNodeByID(pnode->GetXformVolumeNodeID()));

  if (!fixedInputVolume) {
    std::cerr << "Failed to look up fixed volume!" << std::endl;
    return -1;
  }
  if (!movingInputVolume) {
    std::cerr << "Failed to look up moving volume!" << std::endl;
    return -1;
  }
  if (!warpedInputVolume) {
    std::cerr << "Failed to look up warped volume!" << std::endl;
    return -1;
  }
  if (!xformInputVolume) {
    std::cerr << "Failed to look up xform volume!" << std::endl;
    return -1;
  }
#endif

#if 0
  vtkMRMLScalarVolumeNode *refVolume;
  vtkMRMLVolumeNode *outputVolume = NULL;
  vtkMatrix4x4 *inputRASToIJK = vtkMatrix4x4::New();
  vtkMatrix4x4 *inputIJKToRAS = vtkMatrix4x4::New();
  vtkMatrix4x4 *outputRASToIJK = vtkMatrix4x4::New();
  vtkMatrix4x4 *outputIJKToRAS = vtkMatrix4x4::New();
  vtkMRMLLinearTransformNode *movingVolumeTransform = NULL, *roiTransform = NULL;

  // make sure inputs are initialized
  if(!inputVolume || !inputROI )
    {
    std::cerr << "CropVolume: Inputs are not initialized" << std::endl;
    return -1;
    }

  // check the output volume type
  vtkMRMLDiffusionTensorVolumeNode *dtvnode= vtkMRMLDiffusionTensorVolumeNode::SafeDownCast(inputVolume);
  vtkMRMLDiffusionWeightedVolumeNode *dwvnode= vtkMRMLDiffusionWeightedVolumeNode::SafeDownCast(inputVolume);
  vtkMRMLVectorVolumeNode *vvnode= vtkMRMLVectorVolumeNode::SafeDownCast(inputVolume);
  vtkMRMLScalarVolumeNode *svnode = vtkMRMLScalarVolumeNode::SafeDownCast(inputVolume);

  if(!this->Internal->VolumesLogic){
      std::cerr << "CropVolume: ERROR: failed to get hold of Volumes logic" << std::endl;
      return -2;
  }
  if(dtvnode){
    std::cerr << "CropVolume: ERROR: Diffusion tensor volumes are not supported by this module!" << std::endl;
    return -2;
  }
 
  std::ostringstream outSS;
  double outputSpacing[3], spacingScaleConst = pnode->GetSpacingScalingConst();
  outSS << inputVolume->GetName() << "-subvolume-scale_" << spacingScaleConst;

 if(dwvnode){
    outputVolume = vtkMRMLVolumeNode::SafeDownCast(this->Internal->VolumesLogic->CloneVolume(this->GetMRMLScene(), inputVolume, outSS.str().c_str()));
  }
  if(vvnode){
    outputVolume = vtkMRMLVolumeNode::SafeDownCast(this->Internal->VolumesLogic->CloneVolume(this->GetMRMLScene(), inputVolume, outSS.str().c_str()));
  }
  if(svnode){
    outputVolume = vtkMRMLVolumeNode::SafeDownCast(this->Internal->VolumesLogic->CloneVolume(this->GetMRMLScene(), inputVolume, outSS.str().c_str()));
  }
  refVolume = this->Internal->VolumesLogic->CreateLabelVolume(this->GetMRMLScene(), inputVolume, "CropVolume_ref_volume");
  refVolume->HideFromEditorsOn();

  //vtkMatrix4x4 *volumeXform = vtkMatrix4x4::New();
  //vtkMatrix4x4 *roiXform = vtkMatrix4x4::New();
  //vtkMatrix4x4 *T = vtkMatrix4x4::New();

  refVolume->GetRASToIJKMatrix(inputRASToIJK);
  refVolume->GetIJKToRASMatrix(inputIJKToRAS);
  outputRASToIJK->Identity();
  outputIJKToRAS->Identity();

  //T->Identity();
  //roiXform->Identity();
  //volumeXform->Identity();

  // prepare the resampling reference volume
  double roiRadius[3], roiXYZ[3];
  inputROI->GetRadiusXYZ(roiRadius);
  inputROI->GetXYZ(roiXYZ);
  std::cerr << "ROI radius: " << roiRadius[0] << "," << roiRadius[1] << "," << roiRadius[2] << std::endl;
  std::cerr << "ROI center: " << roiXYZ[0] << "," << roiXYZ[1] << "," << roiXYZ[2] << std::endl;

  double* inputSpacing = inputVolume->GetSpacing();
  double minSpacing = inputSpacing[0];
  if (minSpacing > inputSpacing[1])
    {
    minSpacing = inputSpacing[1];
    }
  if (minSpacing > inputSpacing[2])
    {
    minSpacing = inputSpacing[2];
    }

  if(pnode->GetIsotropicResampling()){
      outputSpacing[0] = minSpacing*spacingScaleConst;
      outputSpacing[1] = minSpacing*spacingScaleConst;
      outputSpacing[2] = minSpacing*spacingScaleConst;
  } else {
      outputSpacing[0] = inputSpacing[0]*spacingScaleConst;
      outputSpacing[1] = inputSpacing[1]*spacingScaleConst;
      outputSpacing[2] = inputSpacing[2]*spacingScaleConst;
  }

  int outputExtent[3];

  outputExtent[0] = roiRadius[0]/outputSpacing[0]*2.;
  outputExtent[1] = roiRadius[1]/outputSpacing[1]*2.;
  outputExtent[2] = roiRadius[2]/outputSpacing[2]*2.;

  outputIJKToRAS->SetElement(0,0,outputSpacing[0]);
  outputIJKToRAS->SetElement(1,1,outputSpacing[1]);
  outputIJKToRAS->SetElement(2,2,outputSpacing[2]);

  outputIJKToRAS->SetElement(0,3,roiXYZ[0]-roiRadius[0]+outputSpacing[0]*.5);
  outputIJKToRAS->SetElement(1,3,roiXYZ[1]-roiRadius[1]+outputSpacing[1]*.5);
  outputIJKToRAS->SetElement(2,3,roiXYZ[2]-roiRadius[2]+outputSpacing[2]*.5);

  // account for the ROI parent transform, if present
  roiTransform = vtkMRMLLinearTransformNode::SafeDownCast(inputROI->GetParentTransformNode());
  if(roiTransform){
    vtkMatrix4x4 *roiMatrix = vtkMatrix4x4::New();
    roiTransform->GetMatrixTransformToWorld(roiMatrix);
    outputIJKToRAS->Multiply4x4(roiMatrix, outputIJKToRAS, outputIJKToRAS);
  }

  outputRASToIJK->DeepCopy(outputIJKToRAS);
  outputRASToIJK->Invert();

  vtkImageData* outputImageData = vtkImageData::New();
  outputImageData->SetDimensions(outputExtent[0], outputExtent[1], outputExtent[2]);
  outputImageData->AllocateScalars();

  refVolume->SetAndObserveImageData(outputImageData);
  outputImageData->Delete();

  refVolume->SetIJKToRASMatrix(outputIJKToRAS);
  refVolume->SetRASToIJKMatrix(outputRASToIJK);

  inputRASToIJK->Delete();
  inputIJKToRAS->Delete();
  outputRASToIJK->Delete();
  outputIJKToRAS->Delete();

  if(this->Internal->ResampleLogic == 0)
    {
    std::cerr << "CropVolume: ERROR: resample logic is not set!";
    return -3;
    }

  vtkSmartPointer<vtkMRMLCommandLineModuleNode> cmdNode =
    this->Internal->ResampleLogic->CreateNodeInScene();
  assert(cmdNode.GetPointer() != 0);

  cmdNode->SetParameterAsString("inputVolume", inputVolume->GetID());
  cmdNode->SetParameterAsString("referenceVolume",refVolume->GetID());
  cmdNode->SetParameterAsString("outputVolume",outputVolume->GetID());

  movingVolumeTransform = vtkMRMLLinearTransformNode::SafeDownCast(inputVolume->GetParentTransformNode());

  if(movingVolumeTransform != NULL)
    cmdNode->SetParameterAsString("transformationFile",movingVolumeTransform->GetID());

  std::string interp = "linear";
  switch(pnode->GetInterpolationMode()){
    case 1: interp = "nn"; break;
    case 2: interp = "linear"; break;
    case 3: interp = "ws"; break;
    case 4: interp = "bs"; break;
  }

  cmdNode->SetParameterAsString("interpolationType", interp.c_str());
  this->Internal->ResampleLogic->ApplyAndWait(cmdNode);

  this->GetMRMLScene()->RemoveNode(refVolume);
  this->GetMRMLScene()->RemoveNode(cmdNode);

  outputVolume->SetAndObserveTransformNodeID(NULL);
  outputVolume->ModifiedSinceReadOn();
  pnode->SetOutputVolumeNodeID(outputVolume->GetID());
#endif

  fprintf (stderr, "Apply ()\n");
  return 0;
}
