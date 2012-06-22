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
  Q_DECLARE_PUBLIC(qSlicerPlmSlicerBsplineModuleWidget);
protected:
  qSlicerPlmSlicerBsplineModuleWidget* const q_ptr;
public:
  qSlicerPlmSlicerBsplineModuleWidgetPrivate(qSlicerPlmSlicerBsplineModuleWidget& object);
  vtkSlicerPlmSlicerBsplineLogic* logic() const;
  
};

//-----------------------------------------------------------------------------
// qSlicerPlmSlicerBsplineModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModuleWidgetPrivate::qSlicerPlmSlicerBsplineModuleWidgetPrivate(qSlicerPlmSlicerBsplineModuleWidget& object) : q_ptr(&object)
{
}

//-----------------------------------------------------------------------------
vtkSlicerPlmSlicerBsplineLogic* qSlicerPlmSlicerBsplineModuleWidgetPrivate::logic() const
{
  Q_Q(const qSlicerPlmSlicerBsplineModuleWidget);
  return vtkSlicerPlmSlicerBsplineLogic::SafeDownCast(q->logic());
}


//-----------------------------------------------------------------------------
// qSlicerPlmSlicerBsplineModuleWidget methods

//-----------------------------------------------------------------------------
qSlicerPlmSlicerBsplineModuleWidget::qSlicerPlmSlicerBsplineModuleWidget(QWidget* _parent)
  : Superclass( _parent )
  , d_ptr( new qSlicerPlmSlicerBsplineModuleWidgetPrivate(*this) )
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

  /* Volumes */
  connect (d->fixedImageMRMLNodeComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)),
          this, SLOT(onFixedVolumeChanged()));
  connect (d->movingImageMRMLNodeComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)),
          this, SLOT(onMovingVolumeChanged()));
  connect (d->warpedImageMRMLNodeComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)),
          this, SLOT(onWarpedVolumeChanged()));
  connect (d->xformImageMRMLNodeComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)),
          this, SLOT(onXformVolumeChanged()));

  /* Similarity Metric Radios */
  connect (d->radioButtonMSE, SIGNAL(toggled(bool)),
          this, SLOT(onMSEChanged()));
  connect (d->radioButtonMI, SIGNAL(toggled(bool)),
          this, SLOT(onMIChanged()));

  /* the "GO" button */
  connect (d->registerPushButton, SIGNAL(clicked()),
          this, SLOT(onApply()));
}

void qSlicerPlmSlicerBsplineModuleWidget::enter()
{
  this->onFixedVolumeChanged ();
  this->onMovingVolumeChanged ();
  this->onWarpedVolumeChanged ();
  this->onXformVolumeChanged ();
  this->onMSEChanged ();
  this->onMIChanged ();
}

void qSlicerPlmSlicerBsplineModuleWidget::setMRMLScene(vtkMRMLScene* scene)
{
  this->Superclass::setMRMLScene (scene);
  if(scene == NULL) return;

  /* JAS 2012.06.2012 */
  /* parameter node creation currently segfaults slicer.
   * (See line 137)
   * New macro is not being satisfied. */
//  this->initializeParameterNode (scene);

#if 0
  this->updateWidget();

  // observe close event
  qvtkReconnect(this->mrmlScene(), vtkMRMLScene::EndCloseEvent, 
    this, SLOT(onEndCloseEvent()));
#endif

}

#if 0
//-----------------------------------------------------------------------------
void qSlicerCropVolumeModuleWidget::initializeNode(vtkMRMLNode *n)
{
  vtkMRMLScene* scene = qobject_cast<qMRMLNodeFactory*>(this->sender())->mrmlScene();
  vtkMRMLAnnotationROINode::SafeDownCast(n)->Initialize(scene);
}
#endif

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::initializeParameterNode(vtkMRMLScene* scene)
{
    vtkCollection* parameterNodes = scene->GetNodesByClass("vtkMRMLPlmSlicerBsplineParametersNode");

    if (parameterNodes->GetNumberOfItems() > 0) {
        this->parametersNode = vtkMRMLPlmSlicerBsplineParametersNode::SafeDownCast(parameterNodes->GetItemAsObject(0));
        if (!this->parametersNode) {
            qCritical() << "FATAL ERROR: Cannot instantiate PlmSlicerBsplineParameterNode";
            Q_ASSERT(this->parametersNode);
        }
    } else {
        qDebug() << "No PlmSlicerBspline parameter nodes found!";
        this->parametersNode = vtkMRMLPlmSlicerBsplineParametersNode::New();
        scene->AddNode(this->parametersNode);
        this->parametersNode->Delete();
    }

    parameterNodes->Delete();
}


//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::updateParameters()
{
  Q_D(qSlicerPlmSlicerBsplineModuleWidget);
  if(!this->parametersNode)
    return;
  vtkMRMLPlmSlicerBsplineParametersNode *pNode = this->parametersNode;

  /* Fixed Image */
  vtkMRMLNode *fixedVolumeNode = d->fixedImageMRMLNodeComboBox->currentNode();
  if (fixedVolumeNode) {
    pNode->SetFixedVolumeNodeID(fixedVolumeNode->GetID());
  } else {
    pNode->SetFixedVolumeNodeID(NULL);
  }

  /* Moving Image */
  vtkMRMLNode *movingVolumeNode = d->movingImageMRMLNodeComboBox->currentNode();
  if (movingVolumeNode) {
    pNode->SetMovingVolumeNodeID(movingVolumeNode->GetID());
  } else {
    pNode->SetMovingVolumeNodeID(NULL);
  }

  /* Warped Image */
  vtkMRMLNode *warpedVolumeNode = d->warpedImageMRMLNodeComboBox->currentNode();
  if (warpedVolumeNode) {
    pNode->SetWarpedVolumeNodeID(warpedVolumeNode->GetID());
  } else {
    pNode->SetWarpedVolumeNodeID(NULL);
  }

  /* Vector Field Image */
  vtkMRMLNode *xformVolumeNode = d->xformImageMRMLNodeComboBox->currentNode();
  if (xformVolumeNode) {
    pNode->SetXformVolumeNodeID(xformVolumeNode->GetID());
  } else {
    pNode->SetXformVolumeNodeID(NULL);
  }

//  pNode->SetROIVisibility(d->VisibilityButton->isChecked());

#if 0
  if(d->NNRadioButton->isChecked())
    pNode->SetInterpolationMode(1);
  else if(d->LinearRadioButton->isChecked())
    pNode->SetInterpolationMode(2);
  else if(d->WSRadioButton->isChecked())
    pNode->SetInterpolationMode(3);
  else if(d->BSRadioButton->isChecked())
    pNode->SetInterpolationMode(4);

  if(d->IsotropicCheckbox->isChecked())
    pNode->SetIsotropicResampling(1);

  pNode->SetSpacingScalingConst(d->SpacingScalingSpinBox->value());
#endif
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onApply()
{
  Q_D(const qSlicerPlmSlicerBsplineModuleWidget);

#if 0
  /* GCS 2012-01-21: This seems simpler/better than the CropVolume 
     example code which adds a logic() member to the Q_D class */
  vtkSlicerPlmSlicerBsplineLogic *logic = 
    vtkSlicerPlmSlicerBsplineLogic::SafeDownCast(this->logic());
#endif

  vtkSlicerPlmSlicerBsplineLogic *logic = d->logic();
  fprintf (stderr, "parametersNode: %p\n", this->parametersNode);
//  logic->Apply(this->parametersNode);

#if 0
  vtkIndent ind;
  this->parametersNode->PrintSelf(cout, ind);
#endif
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onFixedVolumeChanged()
{
    Q_D(qSlicerPlmSlicerBsplineModuleWidget);
    if (!this->parametersNode) return;

    vtkMRMLNode* node = d->fixedImageMRMLNodeComboBox->currentNode();
    if (node) {
        this->parametersNode->SetFixedVolumeNodeID(node->GetID());
    }
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onMovingVolumeChanged()
{
    Q_D(qSlicerPlmSlicerBsplineModuleWidget);
    if (!this->parametersNode) return;

    vtkMRMLNode* node = d->movingImageMRMLNodeComboBox->currentNode();
    if (node) {
        this->parametersNode->SetMovingVolumeNodeID(node->GetID());
    }
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onWarpedVolumeChanged()
{
    Q_D(qSlicerPlmSlicerBsplineModuleWidget);
    if (!this->parametersNode) return;

    vtkMRMLNode* node = d->warpedImageMRMLNodeComboBox->currentNode();
    if (node) {
        this->parametersNode->SetWarpedVolumeNodeID(node->GetID());
    }
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onXformVolumeChanged()
{
    Q_D(qSlicerPlmSlicerBsplineModuleWidget);
    if (!this->parametersNode) return;

    vtkMRMLNode* node = d->xformImageMRMLNodeComboBox->currentNode();
    if (node) {
        this->parametersNode->SetXformVolumeNodeID(node->GetID());
    }
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onMSEChanged()
{
    Q_D(qSlicerPlmSlicerBsplineModuleWidget);
    if (!this->parametersNode) return;

    this->parametersNode->SetUseMSE(d->radioButtonMSE->isChecked());
}

//-----------------------------------------------------------------------------
void qSlicerPlmSlicerBsplineModuleWidget::onMIChanged()
{
    Q_D(qSlicerPlmSlicerBsplineModuleWidget);
    if (!this->parametersNode) return;

    this->parametersNode->SetUseMI(d->radioButtonMSE->isChecked());
}

