/*=auto=========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkPlastimatchLoadableLogic.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkPlastimatchLoadableLogic_h
#define __vtkPlastimatchLoadableLogic_h

#include "vtkSlicerModuleLogic.h"
#include "vtkMRMLScene.h"

#include "vtkPlastimatchLoadable.h"
#include "vtkMRMLPlastimatchLoadableNode.h"


class vtkITKGradientAnisotropicDiffusionImageFilter;

class VTK_EXAMPLELOADABLEMODULE_EXPORT vtkPlastimatchLoadableLogic : public vtkSlicerModuleLogic
{
  public:
  static vtkPlastimatchLoadableLogic *New();
  vtkTypeMacro(vtkPlastimatchLoadableLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  // TODO: do we need to observe MRML here?
  virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
                                   void *callData ){};

  // Description: Get/Set MRML node storing parameter values
  vtkGetObjectMacro (PlastimatchLoadableNode, vtkMRMLPlastimatchLoadableNode);
  void SetAndObservePlastimatchLoadableNode(vtkMRMLPlastimatchLoadableNode *n) 
    {
    vtkSetAndObserveMRMLNodeMacro( this->PlastimatchLoadableNode, n);
    }

  // The method that creates and runs VTK or ITK pipeline
  void Apply();
  
protected:
  vtkPlastimatchLoadableLogic();
  virtual ~vtkPlastimatchLoadableLogic();
  vtkPlastimatchLoadableLogic(const vtkPlastimatchLoadableLogic&);
  void operator=(const vtkPlastimatchLoadableLogic&);

  vtkMRMLPlastimatchLoadableNode* PlastimatchLoadableNode;
  vtkITKGradientAnisotropicDiffusionImageFilter* GradientAnisotropicDiffusionImageFilter;

};

#endif

