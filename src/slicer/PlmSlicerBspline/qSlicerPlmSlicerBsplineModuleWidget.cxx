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
#include <QDebug>

// SlicerQt includes
#include "qSlicerPlmSlicerBsplineModuleWidget.h"
#include "ui_qSlicerPlmSlicerBsplineModule.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerPlmSlicerBsplineModuleWidgetPrivate: public Ui_qSlicerPlmSlicerBsplineModule
{
public:
  qSlicerPlmSlicerBsplineModuleWidgetPrivate();
};

//-----------------------------------------------------------------------------
// qSlicerPlmSlicerBsplineModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModuleWidgetPrivate::qSlicerPlmSlicerBsplineModuleWidgetPrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerPlmSlicerBsplineModuleWidget methods

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModuleWidget::qSlicerPlmSlicerBsplineModuleWidget(QWidget* _parent)
  : Superclass( _parent )
  , d_ptr( new qSlicerPlmSlicerBsplineModuleWidgetPrivate )
{
  this->parametersNode = NULL;
}

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModuleWidget::~qSlicerPlmSlicerBsplineModuleWidget()
{
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::setup()
{
  Q_D(qSlicerPlmSlicerBsplineModuleWidget);
  d->setupUi(this);
  this->Superclass::setup();

  connect(d->fixedImageMRMLNodeComboBox, 
      SIGNAL(currentNodeChanged(vtkMRMLNode*)),
      this, SLOT(onInputVolumeChanged()));
  connect(d->movingImageMRMLNodeComboBox, 
      SIGNAL(currentNodeChanged(vtkMRMLNode*)),
      this, SLOT(onInputVolumeChanged()));
}

void qSlicerPlmSlicerBsplineModuleWidget::enter()
{
#if defined (commentout)
  this->onInputVolumeChanged();
  this->onInputROIChanged();
#endif
}

void qSlicerPlmSlicerBsplineModuleWidget::setMRMLScene(vtkMRMLScene* scene){

  this->Superclass::setMRMLScene(scene);
  if(scene == NULL)
    return;

#if defined (commentout)
  vtkCollection* parameterNodes = scene->GetNodesByClass("vtkMRMLPlmSlicerBsplineParametersNode");

  if(parameterNodes->GetNumberOfItems() > 0)
    {
    this->parametersNode = vtkMRMLPlmSlicerBsplineParametersNode::SafeDownCast(parameterNodes->GetItemAsObject(0));
    if(!this->parametersNode)
      {
      qCritical() << "FATAL ERROR: Cannot instantiate PlmSlicerBsplineParameterNode";
      Q_ASSERT(this->parametersNode);
      }
    //InitializeEventListeners(this->parametersNode);
    }
  else
    {
    qDebug() << "No PlmSlicerBspline parameter nodes found!";
    this->parametersNode = vtkMRMLPlmSlicerBsplineParametersNode::New();
    scene->AddNodeNoNotify(this->parametersNode);
    this->parametersNode->Delete();
    }

  parameterNodes->Delete();

  this->updateWidget();

#endif
}
