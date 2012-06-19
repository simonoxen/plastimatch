/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <QDebug>

#include "qSlicerPlmSlicerBsplineModuleWidget.h"
#include "ui_qSlicerPlmSlicerBsplineModule.h"
#include "vtkMRMLPlmSlicerBsplineParametersNode.h"
#include "vtkSlicerPlmSlicerBsplineLogic.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerPlmSlicerBsplineModuleWidgetPrivate
  : public Ui_qSlicerPlmSlicerBsplineModule
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

  connect (d->fixedImageMRMLNodeComboBox, 
          SIGNAL(currentNodeChanged(vtkMRMLNode*)),
          this,
          SLOT(onInputVolumeChanged()));

  connect (d->movingImageMRMLNodeComboBox, 
          SIGNAL(currentNodeChanged(vtkMRMLNode*)),
          this,
          SLOT(onInputVolumeChanged()));

  connect (d->registerPushButton,
          SIGNAL(clicked()),
          this,
          SLOT(onApply()));
}

void qSlicerPlmSlicerBsplineModuleWidget::enter()
{
#if 0
  this->onInputVolumeChanged();
  this->onInputROIChanged();
#endif
}

void qSlicerPlmSlicerBsplineModuleWidget::setMRMLScene(vtkMRMLScene* scene)
{
  this->Superclass::setMRMLScene(scene);
  if(scene == NULL)
    return;

#if 0
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

void qSlicerPlmSlicerBsplineModuleWidget::onApply()
{
  Q_D(const qSlicerPlmSlicerBsplineModuleWidget);

  /* GCS 2012-01-21: This seems simpler/better than the CropVolume 
     example code which adds a logic() member to the Q_D class */
  vtkSlicerPlmSlicerBsplineLogic *logic = 
    vtkSlicerPlmSlicerBsplineLogic::SafeDownCast(this->logic());

#if 0
  vtkSlicerPlmSlicerBsplineLogic *logic = d->logic();
  if(!logic->Apply(this->parametersNode))
    {
    std::cerr << "Propagating to the selection node" << std::endl;
    vtkSlicerApplicationLogic *appLogic = this->module()->appLogic();
    vtkMRMLSelectionNode *selectionNode = appLogic->GetSelectionNode();
    selectionNode->SetReferenceActiveVolumeID (
      this->parametersNode->GetOutputVolumeNodeID());
    appLogic->PropagateVolumeSelection();
    }
#endif
}
