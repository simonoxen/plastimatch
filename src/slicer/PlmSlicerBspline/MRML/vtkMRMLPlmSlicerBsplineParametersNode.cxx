/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
// VTK includes
#include <vtkCommand.h>
#include <vtkObjectFactory.h>

// MRML includes
#include "vtkMRMLVolumeNode.h"

// CropModuleMRML includes
#include "vtkMRMLPlmSlicerBsplineParametersNode.h"

// AnnotationModuleMRML includes
#include "vtkMRMLAnnotationROINode.h"

// STD includes

//----------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLPlmSlicerBsplineParametersNode);

//----------------------------------------------------------------------------
vtkMRMLPlmSlicerBsplineParametersNode::vtkMRMLPlmSlicerBsplineParametersNode()
{
  this->HideFromEditors = 1;

  this->InputVolumeNodeID = NULL;
  this->InputVolumeNode = NULL;

  this->OutputVolumeNodeID = NULL;
  this->OutputVolumeNode = NULL;

  this->ROINodeID = NULL;
  this->ROINode = NULL;

  this->ROIVisibility = false;
  this->InterpolationMode = 2;

  this->SpacingScalingConst = 1.;
}

//----------------------------------------------------------------------------
vtkMRMLPlmSlicerBsplineParametersNode::~vtkMRMLPlmSlicerBsplineParametersNode()
{
  if (this->InputVolumeNodeID)
    {
    this->SetAndObserveInputVolumeNodeID(NULL);
    }

  if (this->OutputVolumeNodeID)
    {
    this->SetAndObserveOutputVolumeNodeID(NULL);
    }

  if (this->ROINodeID)
    {
    this->SetAndObserveROINodeID(NULL);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::ReadXMLAttributes(const char** atts)
{
  std::cerr << "Reading PlmSlicerBspline param node!" << std::endl;
  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL)
  {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "inputVolumeNodeID"))
    {
      this->SetInputVolumeNodeID(attValue);
      continue;
    }
    if (!strcmp(attName, "outputVolumeNodeID"))
    {
      this->SetOutputVolumeNodeID(attValue);
      continue;
    }
    if (!strcmp(attName, "ROINodeID"))
    {
      this->SetROINodeID(attValue);
      continue;
    }
    if (!strcmp(attName,"ROIVisibility"))
    {
      std::stringstream ss;
      ss << attValue;
      ss >> this->ROIVisibility;
      continue;
    }
    if (!strcmp(attName,"interpolationMode"))
    {
      std::stringstream ss;
      ss << attValue;
      ss >> this->InterpolationMode;
      continue;
    }
  }

  this->WriteXML(std::cout,1);
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  of << indent << " inputVolumeNodeID=\"" << (this->InputVolumeNodeID ? this->InputVolumeNodeID : "NULL") << "\"";
  of << indent << " outputVolumeNodeID=\"" << (this->OutputVolumeNodeID ? this->OutputVolumeNodeID : "NULL") << "\"";
  of << indent << " ROIVisibility=\""<< this->ROIVisibility << "\"";
  of << indent << " ROINodeID=\"" << (this->ROINodeID ? this->ROINodeID : "NULL") << "\"";
  of << indent << " interpolationMode=\"" << this->InterpolationMode << "\"";
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::UpdateReferenceID(const char *oldID, const char *newID)
{
  if (this->InputVolumeNodeID && !strcmp(oldID, this->InputVolumeNodeID))
    {
    this->SetAndObserveInputVolumeNodeID(newID);
    }
  if (this->OutputVolumeNodeID && !strcmp(oldID, this->OutputVolumeNodeID))
    {
    this->SetAndObserveOutputVolumeNodeID(newID);
    }
  if (this->ROINodeID && !strcmp(oldID, this->ROINodeID))
    {
    this->SetAndObserveROINodeID(newID);
    }
}

//-----------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::UpdateReferences()
{
   Superclass::UpdateReferences();

  if (this->InputVolumeNodeID != NULL && this->Scene->GetNodeByID(this->InputVolumeNodeID) == NULL)
    {
    this->SetAndObserveInputVolumeNodeID(NULL);
    }
  if (this->OutputVolumeNodeID != NULL && this->Scene->GetNodeByID(this->OutputVolumeNodeID) == NULL)
    {
    this->SetAndObserveOutputVolumeNodeID(NULL);
    }
  if (this->ROINodeID != NULL && this->Scene->GetNodeByID(this->ROINodeID) == NULL)
    {
    this->SetAndObserveROINodeID(NULL);
    }
}

//----------------------------------------------------------------------------
// Copy the node\"s attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, SliceID
void vtkMRMLPlmSlicerBsplineParametersNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLPlmSlicerBsplineParametersNode *node = vtkMRMLPlmSlicerBsplineParametersNode::SafeDownCast(anode);
  this->DisableModifiedEventOn();

  this->SetInputVolumeNodeID(node->GetInputVolumeNodeID());
  this->SetOutputVolumeNodeID(node->GetOutputVolumeNodeID());
  this->SetROINodeID(node->GetROINodeID());
  this->SetInterpolationMode(node->GetInterpolationMode());
  this->SetROIVisibility(node->GetROIVisibility());
  
  this->DisableModifiedEventOff();
  this->InvokePendingModifiedEvent();
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::SetAndObserveInputVolumeNodeID(const char *volumeNodeID)
{
  vtkSetAndObserveMRMLObjectMacro(this->InputVolumeNode, NULL);

  if (volumeNodeID != NULL)
  {
    this->SetInputVolumeNodeID(volumeNodeID);
    vtkMRMLVolumeNode *node = this->GetInputVolumeNode();
    vtkSetAndObserveMRMLObjectMacro(this->InputVolumeNode, node);
  }
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::SetAndObserveOutputVolumeNodeID(const char *volumeNodeID)
{
  vtkSetAndObserveMRMLObjectMacro(this->OutputVolumeNode, NULL);

  if (volumeNodeID != NULL)
  {
    this->SetOutputVolumeNodeID(volumeNodeID);
    vtkMRMLVolumeNode *node = this->GetOutputVolumeNode();
    vtkSetAndObserveMRMLObjectMacro(this->OutputVolumeNode, node);
  }
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::SetAndObserveROINodeID(const char *ROINodeID)
{
  vtkSetAndObserveMRMLObjectMacro(this->ROINode, NULL);

  if (ROINodeID != NULL)
  {
    this->SetROINodeID(ROINodeID);
    vtkMRMLAnnotationROINode *node = this->GetROINode();
    vtkSetAndObserveMRMLObjectMacro(this->ROINode, node);
  }
}

//----------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkMRMLPlmSlicerBsplineParametersNode::GetInputVolumeNode()
{
  if (this->InputVolumeNodeID == NULL)
    {
    vtkSetAndObserveMRMLObjectMacro(this->InputVolumeNode, NULL);
    }
  else if (this->GetScene() &&
           ((this->InputVolumeNode != NULL && strcmp(this->InputVolumeNode->GetID(), this->InputVolumeNodeID)) ||
            (this->InputVolumeNode == NULL)) )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->InputVolumeNodeID);
    vtkSetAndObserveMRMLObjectMacro(this->InputVolumeNode, vtkMRMLVolumeNode::SafeDownCast(snode));
    }
  return this->InputVolumeNode;
}

//----------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkMRMLPlmSlicerBsplineParametersNode::GetOutputVolumeNode()
{
  if (this->OutputVolumeNodeID == NULL)
    {
    vtkSetAndObserveMRMLObjectMacro(this->OutputVolumeNode, NULL);
    }
  else if (this->GetScene() &&
           ((this->OutputVolumeNode != NULL && strcmp(this->OutputVolumeNode->GetID(), this->OutputVolumeNodeID)) ||
            (this->OutputVolumeNode == NULL)) )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->OutputVolumeNodeID);
    vtkSetAndObserveMRMLObjectMacro(this->OutputVolumeNode, vtkMRMLVolumeNode::SafeDownCast(snode));
    }
  return this->OutputVolumeNode;
}

//----------------------------------------------------------------------------
vtkMRMLAnnotationROINode* vtkMRMLPlmSlicerBsplineParametersNode::GetROINode()
{
  if (this->ROINodeID == NULL)
    {
    vtkSetAndObserveMRMLObjectMacro(this->ROINode, NULL);
    }
  else if (this->GetScene() &&
           ((this->ROINode != NULL && strcmp(this->ROINode->GetID(), this->ROINodeID)) ||
            (this->ROINode == NULL)) )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->ROINodeID);
    vtkSetAndObserveMRMLObjectMacro(this->ROINode, vtkMRMLAnnotationROINode::SafeDownCast(snode));
    }
  return this->ROINode;
}

//-----------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::UpdateScene(vtkMRMLScene *scene)
{
  Superclass::UpdateScene(scene);
  this->SetAndObserveInputVolumeNodeID(this->InputVolumeNodeID);
  this->SetAndObserveOutputVolumeNodeID(this->OutputVolumeNodeID);
  this->SetAndObserveROINodeID(this->ROINodeID);
}

//---------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::ProcessMRMLEvents ( vtkObject *caller,
                                                    unsigned long event,
                                                    void *callData )
{
    Superclass::ProcessMRMLEvents(caller, event, callData);
    this->InvokeEvent(vtkCommand::ModifiedEvent, NULL);
    return;
}

//----------------------------------------------------------------------------
void vtkMRMLPlmSlicerBsplineParametersNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);

  os << "InputVolumeNodeID: " << ( (this->InputVolumeNodeID) ? this->InputVolumeNodeID : "None" ) << "\n";
  os << "OutputVolumeNodeID: " << ( (this->OutputVolumeNodeID) ? this->OutputVolumeNodeID : "None" ) << "\n";
  os << "ROINodeID: " << ( (this->ROINodeID) ? this->ROINodeID : "None" ) << "\n";
  os << "ROIVisibility: " << this->ROIVisibility << "\n";
  os << "InterpolationMode: " << this->InterpolationMode << "\n";
  os << "IsotropicResampling: " << this->IsotropicResampling << "\n";
}

// End
