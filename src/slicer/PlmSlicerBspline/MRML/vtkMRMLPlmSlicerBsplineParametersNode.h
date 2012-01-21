/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __vtkMRMLPlmSlicerBsplineParametersNode_h
#define __vtkMRMLPlmSlicerBsplineParametersNode_h

#include "vtkMRML.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLNode.h"
#include "vtkSlicerPlmSlicerBsplineModuleMRMLExport.h"

class vtkMRMLAnnotationROINode;
class vtkMRMLVolumeNode;

/// \ingroup Slicer_QtModules_PlmSlicerBspline
class VTK_SLICER_PLMSLICERBSPLINE_MODULE_MRML_EXPORT vtkMRMLPlmSlicerBsplineParametersNode : public vtkMRMLNode
{
  public:   

  static vtkMRMLPlmSlicerBsplineParametersNode *New();
  vtkTypeMacro(vtkMRMLPlmSlicerBsplineParametersNode,vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);

  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "PlmSlicerBsplineParameters";};

  // Description:
  // Update the stored reference to another node in the scene
  virtual void UpdateReferenceID(const char *oldID, const char *newID);

  // Description:
  // Updates this node if it depends on other nodes
  // when the node is deleted in the scene
  virtual void UpdateReferences();

  // Description:
  virtual void UpdateScene(vtkMRMLScene *scene);

  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData);

  // Description:
  vtkGetStringMacro (InputVolumeNodeID);
  void SetAndObserveInputVolumeNodeID(const char *volumeNodeID);
  vtkMRMLVolumeNode* GetInputVolumeNode();
  
  vtkGetStringMacro (OutputVolumeNodeID);
  void SetAndObserveOutputVolumeNodeID(const char *volumeNodeID);
  vtkMRMLVolumeNode* GetOutputVolumeNode();

  vtkGetStringMacro (ROINodeID);
  void SetAndObserveROINodeID(const char *ROINodeID);
  vtkMRMLAnnotationROINode* GetROINode();

  vtkSetMacro(IsotropicResampling,bool);
  vtkGetMacro(IsotropicResampling,bool);
  vtkBooleanMacro(IsotropicResampling,bool);

  vtkSetMacro(ROIVisibility,bool);
  vtkGetMacro(ROIVisibility,bool);
  vtkBooleanMacro(ROIVisibility,bool);

  typedef enum {NearestNeighbor, Linear, Cubic}
   InterpolationModeType;

  vtkSetMacro(InterpolationMode, int);
  vtkGetMacro(InterpolationMode, int);

  vtkSetMacro(SpacingScalingConst, double);
  vtkGetMacro(SpacingScalingConst, double);

protected:
  vtkMRMLPlmSlicerBsplineParametersNode();
  ~vtkMRMLPlmSlicerBsplineParametersNode();
  vtkMRMLPlmSlicerBsplineParametersNode(const vtkMRMLPlmSlicerBsplineParametersNode&);
  void operator=(const vtkMRMLPlmSlicerBsplineParametersNode&);

  char *InputVolumeNodeID;
  char *OutputVolumeNodeID;

  vtkSetReferenceStringMacro(InputVolumeNodeID);
  vtkSetReferenceStringMacro(OutputVolumeNodeID);

  vtkMRMLVolumeNode* InputVolumeNode;
  vtkMRMLVolumeNode* OutputVolumeNode;

  char *ROINodeID;
  
  vtkSetReferenceStringMacro(ROINodeID);

  vtkMRMLAnnotationROINode *ROINode;

  bool ROIVisibility;
  int InterpolationMode;
  bool IsotropicResampling;
  double SpacingScalingConst;
};

#endif

