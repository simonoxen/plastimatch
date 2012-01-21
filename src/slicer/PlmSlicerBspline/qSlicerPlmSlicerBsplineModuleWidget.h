/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __qSlicerPlmSlicerBsplineModuleWidget_h
#define __qSlicerPlmSlicerBsplineModuleWidget_h

#include <qSlicerAbstractModuleWidget.h>

#include "qSlicerPlmSlicerBsplineModuleExport.h"

class qSlicerPlmSlicerBsplineModuleWidgetPrivate;
class vtkMRMLNode;
class vtkMRMLPlmSlicerBsplineParametersNode;

/// \ingroup Slicer_QtModules_ExtensionTemplate
class Q_SLICER_QTMODULES_PLMSLICERBSPLINE_EXPORT qSlicerPlmSlicerBsplineModuleWidget :
  public qSlicerAbstractModuleWidget
{
  Q_OBJECT

public:

  typedef qSlicerAbstractModuleWidget Superclass;
  qSlicerPlmSlicerBsplineModuleWidget(QWidget *parent=0);
  virtual ~qSlicerPlmSlicerBsplineModuleWidget();

public slots:


protected:
  QScopedPointer<qSlicerPlmSlicerBsplineModuleWidgetPrivate> d_ptr;
  
  virtual void setup();
  virtual void enter();
  virtual void setMRMLScene(vtkMRMLScene*);

protected slots:
  void onApply();

private:
  Q_DECLARE_PRIVATE(qSlicerPlmSlicerBsplineModuleWidget);
  Q_DISABLE_COPY(qSlicerPlmSlicerBsplineModuleWidget);

private:
  vtkMRMLPlmSlicerBsplineParametersNode *parametersNode;

};

#endif
