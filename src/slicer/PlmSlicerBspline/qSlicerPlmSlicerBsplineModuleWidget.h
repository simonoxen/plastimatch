/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

#ifndef __qSlicerPlmSlicerBsplineModuleWidget_h
#define __qSlicerPlmSlicerBsplineModuleWidget_h

// SlicerQt includes
#include "qSlicerAbstractModuleWidget.h"

#include "qSlicerPlmSlicerBsplineModuleExport.h"

class qSlicerPlmSlicerBsplineModuleWidgetPrivate;
class vtkMRMLNode;

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

private:
  Q_DECLARE_PRIVATE(qSlicerPlmSlicerBsplineModuleWidget);
  Q_DISABLE_COPY(qSlicerPlmSlicerBsplineModuleWidget);
};

#endif
