/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __vtkSlicerPlmSlicerBsplineLogic_h
#define __vtkSlicerPlmSlicerBsplineLogic_h

#include <cstdlib>
#include <vtkSlicerModuleLogic.h>

#include "vtkSlicerPlmSlicerBsplineModuleLogicExport.h"

/// \ingroup Slicer_QtModules_ExtensionTemplate
class VTK_SLICER_PLMSLICERBSPLINE_MODULE_LOGIC_EXPORT vtkSlicerPlmSlicerBsplineLogic :
  public vtkSlicerModuleLogic
{
public:

  static vtkSlicerPlmSlicerBsplineLogic *New();
  vtkTypeMacro(vtkSlicerPlmSlicerBsplineLogic, vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// Initialize listening to MRML events
  void InitializeEventListeners();

protected:
  vtkSlicerPlmSlicerBsplineLogic();
  virtual ~vtkSlicerPlmSlicerBsplineLogic();

  /// Register MRML Node classes to Scene. Gets called automatically when the MRMLScene is attached to this logic class.
  virtual void RegisterNodes();
  virtual void UpdateFromMRMLScene();
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node);
  virtual void OnMRMLSceneNodeRemoved(vtkMRMLNode* node);
private:

  vtkSlicerPlmSlicerBsplineLogic(const vtkSlicerPlmSlicerBsplineLogic&); // Not implemented
  void operator=(const vtkSlicerPlmSlicerBsplineLogic&);               // Not implemented
};

#endif