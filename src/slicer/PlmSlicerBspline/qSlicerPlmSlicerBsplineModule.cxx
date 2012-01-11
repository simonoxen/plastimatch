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

// Qt includes
#include <QtPlugin>

// ExtensionTemplate Logic includes
#include <vtkSlicerPlmSlicerBsplineLogic.h>

// ExtensionTemplate includes
#include "qSlicerPlmSlicerBsplineModule.h"
#include "qSlicerPlmSlicerBsplineModuleWidget.h"

//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(qSlicerPlmSlicerBsplineModule, qSlicerPlmSlicerBsplineModule);

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerPlmSlicerBsplineModulePrivate
{
public:
  qSlicerPlmSlicerBsplineModulePrivate();
};

//-----------------------------------------------------------------------------
// qSlicerPlmSlicerBsplineModulePrivate methods

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModulePrivate::qSlicerPlmSlicerBsplineModulePrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerPlmSlicerBsplineModule methods

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModule::qSlicerPlmSlicerBsplineModule(QObject* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerPlmSlicerBsplineModulePrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModule::~qSlicerPlmSlicerBsplineModule()
{
}

//-----------------------------------------------------------------------------
QString qSlicerPlmSlicerBsplineModule::helpText()const
{
  return "This PlmSlicerBspline module illustrates how a loadable module should "
      "be implemented.";
}

//-----------------------------------------------------------------------------
QString qSlicerPlmSlicerBsplineModule::acknowledgementText()const
{
  return "This work was supported by ...";
}

//-----------------------------------------------------------------------------
QIcon qSlicerPlmSlicerBsplineModule::icon()const
{
  return QIcon(":/Icons/PlmSlicerBspline.png");
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModule::setup()
{
  this->Superclass::setup();
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleRepresentation * qSlicerPlmSlicerBsplineModule::createWidgetRepresentation()
{
  return new qSlicerPlmSlicerBsplineModuleWidget;
}

//-----------------------------------------------------------------------------
vtkMRMLAbstractLogic* qSlicerPlmSlicerBsplineModule::createLogic()
{
  return vtkSlicerPlmSlicerBsplineLogic::New();
}
