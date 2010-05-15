/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkPlastimatchLoadableGUI.h"
#include "vtkCommand.h"
#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWRadioButton.h"
#include "vtkKWRadioButtonSetWithLabel.h"
#include "vtkKWScale.h"
#include "vtkKWMenu.h"
#include "vtkKWEntry.h"
#include "vtkKWFrame.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkKWPushButton.h"

//------------------------------------------------------------------------------
vtkPlastimatchLoadableGUI* vtkPlastimatchLoadableGUI::New()
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance (
	"vtkPlastimatchLoadableGUI");
    if (ret) {
	return (vtkPlastimatchLoadableGUI*) ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkPlastimatchLoadableGUI;
}


//----------------------------------------------------------------------------
vtkPlastimatchLoadableGUI::vtkPlastimatchLoadableGUI()
{
    this->ConductanceScale = vtkKWScaleWithEntry::New();
    this->TimeStepScale = vtkKWScaleWithEntry::New();
    this->NumberOfIterationsScale = vtkKWScaleWithEntry::New();
    this->FixedVolumeSelector = vtkSlicerNodeSelectorWidget::New();
    this->MovingVolumeSelector = vtkSlicerNodeSelectorWidget::New();
    this->OutVolumeSelector = vtkSlicerNodeSelectorWidget::New();
    this->GADNodeSelector = vtkSlicerNodeSelectorWidget::New();
    this->CostFunctionButtonSet = vtkKWRadioButtonSetWithLabel::New();
    this->ApplyButton = vtkKWPushButton::New();

    this->Logic = NULL;
    this->PlastimatchLoadableNode = NULL;
}

//----------------------------------------------------------------------------
vtkPlastimatchLoadableGUI::~vtkPlastimatchLoadableGUI()
{
    
    if ( this->ConductanceScale ) {
        this->ConductanceScale->SetParent(NULL);
        this->ConductanceScale->Delete();
        this->ConductanceScale = NULL;
    }
    if ( this->TimeStepScale ) {
        this->TimeStepScale->SetParent(NULL);
        this->TimeStepScale->Delete();
        this->TimeStepScale = NULL;
    }
    if ( this->NumberOfIterationsScale ) {
        this->NumberOfIterationsScale->SetParent(NULL);
        this->NumberOfIterationsScale->Delete();
        this->NumberOfIterationsScale = NULL;
    }
    if ( this->FixedVolumeSelector ) {
        this->FixedVolumeSelector->SetParent(NULL);
        this->FixedVolumeSelector->Delete();
        this->FixedVolumeSelector = NULL;
    }
    if ( this->MovingVolumeSelector ) {
        this->MovingVolumeSelector->SetParent(NULL);
        this->MovingVolumeSelector->Delete();
        this->MovingVolumeSelector = NULL;
    }
    if ( this->OutVolumeSelector ) {
        this->OutVolumeSelector->SetParent(NULL);
        this->OutVolumeSelector->Delete();
        this->OutVolumeSelector = NULL;
    }
    if ( this->GADNodeSelector ) {
        this->GADNodeSelector->SetParent(NULL);
        this->GADNodeSelector->Delete();
        this->GADNodeSelector = NULL;
    }
    if (this->CostFunctionButtonSet) {
        this->CostFunctionButtonSet->SetParent(NULL);
        this->CostFunctionButtonSet->Delete();
        this->CostFunctionButtonSet = NULL;
    }
    if ( this->ApplyButton ) {
        this->ApplyButton->SetParent(NULL);
        this->ApplyButton->Delete();
        this->ApplyButton = NULL;
    }

    this->SetLogic (NULL);
    vtkSetMRMLNodeMacro(this->PlastimatchLoadableNode, NULL);

}

//----------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  
}

//---------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::AddGUIObservers ( ) 
{
    this->ConductanceScale->AddObserver (vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ConductanceScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->TimeStepScale->AddObserver (vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->TimeStepScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->NumberOfIterationsScale->AddObserver (vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NumberOfIterationsScale->AddObserver (
	vtkKWScale::ScaleValueChangedEvent, 
	(vtkCommand *)this->GUICallbackCommand);

    this->FixedVolumeSelector->AddObserver (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
    this->MovingVolumeSelector->AddObserver (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
    this->OutVolumeSelector->AddObserver (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
    this->GADNodeSelector->AddObserver (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
    this->CostFunctionButtonSet->GetWidget()->GetWidget(0)->AddObserver (
	vtkKWRadioButton::SelectedStateChangedEvent,
	(vtkCommand *)this->GUICallbackCommand);
    this->CostFunctionButtonSet->GetWidget()->GetWidget(1)->AddObserver (
	vtkKWRadioButton::SelectedStateChangedEvent,
	(vtkCommand *)this->GUICallbackCommand);

    this->ApplyButton->AddObserver (vtkKWPushButton::InvokedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
}


//---------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::RemoveGUIObservers ( )
{
    this->ConductanceScale->RemoveObservers (vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ConductanceScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->TimeStepScale->RemoveObservers (vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->TimeStepScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->NumberOfIterationsScale->RemoveObservers (vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NumberOfIterationsScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->FixedVolumeSelector->RemoveObservers (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
    this->MovingVolumeSelector->RemoveObservers (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
    this->OutVolumeSelector->RemoveObservers (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);

    this->GADNodeSelector->RemoveObservers (
	vtkSlicerNodeSelectorWidget::NodeSelectedEvent, 
	(vtkCommand *)this->GUICallbackCommand);

    this->CostFunctionButtonSet->GetWidget()->GetWidget(0)->RemoveObservers (
	vtkKWRadioButton::SelectedStateChangedEvent,
	(vtkCommand *)this->GUICallbackCommand);
    this->CostFunctionButtonSet->GetWidget()->GetWidget(1)->RemoveObservers (
	vtkKWRadioButton::SelectedStateChangedEvent,
	(vtkCommand *)this->GUICallbackCommand);

    this->ApplyButton->RemoveObservers (
	vtkKWPushButton::InvokedEvent, 
	(vtkCommand *)this->GUICallbackCommand);
}


//---------------------------------------------------------------------------
void 
vtkPlastimatchLoadableGUI::ProcessGUIEvents (
    vtkObject *caller,
    unsigned long event,
    void *callData
)
{
    vtkKWScaleWithEntry *s = vtkKWScaleWithEntry::SafeDownCast(caller);
    vtkKWMenu *v = vtkKWMenu::SafeDownCast(caller);
    vtkKWPushButton *b = vtkKWPushButton::SafeDownCast(caller);
    vtkSlicerNodeSelectorWidget *selector 
	= vtkSlicerNodeSelectorWidget::SafeDownCast(caller);

    if (s == this->ConductanceScale 
	&& event == vtkKWScale::ScaleValueChangedEvent)
    {
	this->UpdateMRML();
    }
    else if (s == this->TimeStepScale 
	&& event == vtkKWScale::ScaleValueChangedEvent)
    {
	this->UpdateMRML();
    }
    else if (s == this->NumberOfIterationsScale 
	&& event == vtkKWScale::ScaleValueChangedEvent)
    {
	this->UpdateMRML();
    }
    else if (selector == this->OutVolumeSelector 
	&& event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent 
	&& this->OutVolumeSelector->GetSelected() != NULL)
    {
	this->UpdateMRML();
    }
    else if (selector == this->FixedVolumeSelector 
	&& event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent 
	&& this->FixedVolumeSelector->GetSelected() != NULL)
    {
	this->UpdateMRML();
    }
    else if (selector == this->MovingVolumeSelector 
	&& event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent 
	&& this->MovingVolumeSelector->GetSelected() != NULL)
    {
	this->UpdateMRML();
    }
    else if (selector == this->GADNodeSelector 
	&& event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent 
	&& this->GADNodeSelector->GetSelected() != NULL)
    {
	vtkMRMLPlastimatchLoadableNode* n 
	    = vtkMRMLPlastimatchLoadableNode::SafeDownCast(
		this->GADNodeSelector->GetSelected());
	this->Logic->SetAndObservePlastimatchLoadableNode(n);
	vtkSetAndObserveMRMLNodeMacro( this->PlastimatchLoadableNode, n);
	this->UpdateGUI();
    }

#if defined (commentout)
    else if ((this->CostFunctionButtonSet->GetWidget()->GetWidget(0)
	    == vtkKWRadioButton::SafeDownCast(caller))
	&& event == vtkKWRadioButton::SelectedStateChangedEvent
	&& (this->CostFunctionButtonSet->GetWidget()
	    ->GetWidget(0)->GetSelectedState() == 1))
    {
	int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
	if (selected >= 0 && selected < (int)this->ConnectorNodeList.size())
	{
	    vtkMRMLPlastimatchLoadableNode* n 
		= vtkMRMLPlastimatchLoadableNode::SafeDownCast(
		    = vtkMRMLIGTLConnectorNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->ConnectorNodeList[selected]));
	    if (connector)
	    {
		connector->SetCostFunction(1);
	    }
	}
    }
    else if ((this->CostFunctionButtonSet->GetWidget()->GetWidget(1)
	    == vtkKWRadioButton::SafeDownCast(caller))
	&& event == vtkKWRadioButton::SelectedStateChangedEvent
	&& (this->CostFunctionButtonSet->GetWidget()
	    ->GetWidget(1)->GetSelectedState() == 1))
    {
	int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
	if (selected >= 0 && selected < (int)this->ConnectorNodeList.size())
	{
	    vtkMRMLIGTLConnectorNode* connector
		= vtkMRMLIGTLConnectorNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->ConnectorNodeList[selected]));
	    if (connector)
	    {
		connector->SetCostFunction(0);
	    }
	}
    }
#endif

    else if (b == this->ApplyButton 
	&& event == vtkKWPushButton::InvokedEvent)
    {
	this->UpdateMRML();
	this->Logic->Apply();
    }
}


//---------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::UpdateMRML ()
{
    vtkMRMLPlastimatchLoadableNode* n = this->GetPlastimatchLoadableNode();
    if (n == NULL)
    {
	// no parameter node selected yet, create new
	this->GADNodeSelector->SetSelectedNew("vtkMRMLPlastimatchLoadableNode");
	this->GADNodeSelector->ProcessNewNodeCommand(
	    "vtkMRMLPlastimatchLoadableNode", "GADParameters");
	n = vtkMRMLPlastimatchLoadableNode::SafeDownCast(
	    this->GADNodeSelector->GetSelected());

	// set an observe new node in Logic
	this->Logic->SetAndObservePlastimatchLoadableNode(n);
	vtkSetAndObserveMRMLNodeMacro(this->PlastimatchLoadableNode, n);
    }

    // save node parameters for Undo
    this->GetLogic()->GetMRMLScene()->SaveStateForUndo(n);

    // set node parameters from GUI widgets
    n->SetConductance(this->ConductanceScale->GetValue());
  
    n->SetTimeStep(this->TimeStepScale->GetValue());
  
    n->SetNumberOfIterations (
	(int) floor (this->NumberOfIterationsScale->GetValue()));
  
    if (this->FixedVolumeSelector->GetSelected() != NULL) {
	n->SetFixedVolumeRef (
	    this->FixedVolumeSelector->GetSelected()->GetID());
    }
    if (this->MovingVolumeSelector->GetSelected() != NULL) {
	n->SetMovingVolumeRef (
	    this->MovingVolumeSelector->GetSelected()->GetID());
    }
    if (this->OutVolumeSelector->GetSelected() != NULL) {
	n->SetOutputVolumeRef (this->OutVolumeSelector->GetSelected()->GetID());
    }
}

//---------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::UpdateGUI ()
{
    vtkMRMLPlastimatchLoadableNode* n = this->GetPlastimatchLoadableNode();
    if (n != NULL) {
	// set GUI widgest from parameter node
	this->ConductanceScale->SetValue(n->GetConductance());
	this->TimeStepScale->SetValue(n->GetTimeStep());
	this->NumberOfIterationsScale->SetValue(n->GetNumberOfIterations());
    }
}

//---------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::ProcessMRMLEvents (
    vtkObject *caller,
    unsigned long event,
    void *callData
)
{
    // if parameter node has been changed externally, update GUI widgets 
    // with new values
    vtkMRMLPlastimatchLoadableNode* node 
	= vtkMRMLPlastimatchLoadableNode::SafeDownCast(caller);
    if (node != NULL && this->GetPlastimatchLoadableNode() == node)
    {
	this->UpdateGUI();
    }
}


//---------------------------------------------------------------------------
void vtkPlastimatchLoadableGUI::BuildGUI ( ) 
{
    vtkSlicerApplication *app 
	= (vtkSlicerApplication *) this->GetApplication();
    vtkMRMLPlastimatchLoadableNode* gadNode 
	= vtkMRMLPlastimatchLoadableNode::New();
    this->Logic->GetMRMLScene()->RegisterNodeClass (gadNode);
    gadNode->Delete();

    this->UIPanel->AddPage ("PlastimatchLoadable", 
	"PlastimatchLoadable", NULL);
    // ---
    // MODULE GUI FRAME 
    // ---
    // Define your help text and build the help frame here.
    const char *help = "The PlastimatchLoadable module....";
    const char *about = 
	"This work was supported by NA-MIC, NAC, BIRN, "
	"NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> "
	"for details. ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "PlastimatchLoadable" );
    this->BuildHelpAndAboutFrame ( page, help, about );
    
    vtkSlicerModuleCollapsibleFrame *moduleFrame 
	= vtkSlicerModuleCollapsibleFrame::New ();
    moduleFrame->SetParent (this->UIPanel->GetPageWidget (
	    "PlastimatchLoadable"));
    moduleFrame->Create ( );
    moduleFrame->SetLabelText ("Gradient Anisotropic Diffusion Filter");
    moduleFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
	moduleFrame->GetWidgetName(), this->UIPanel->GetPageWidget("PlastimatchLoadable")->GetWidgetName());
  
    this->GADNodeSelector->SetNodeClass("vtkMRMLPlastimatchLoadableNode", NULL, NULL, "GADParameters");
    this->GADNodeSelector->SetNewNodeEnabled(1);
    this->GADNodeSelector->NoneEnabledOn();
    this->GADNodeSelector->SetShowHidden(1);
    this->GADNodeSelector->SetParent( moduleFrame->GetFrame() );
    this->GADNodeSelector->Create();
    this->GADNodeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
    this->GADNodeSelector->UpdateMenu();

    this->GADNodeSelector->SetBorderWidth(2);
    this->GADNodeSelector->SetLabelText( "GAD Parameters");
    this->GADNodeSelector->SetBalloonHelpString(
	"select a GAD node from the current mrml scene.");
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->GADNodeSelector->GetWidgetName());

    this->ConductanceScale->SetParent( moduleFrame->GetFrame() );
    this->ConductanceScale->SetLabelText("Conductance");
    this->ConductanceScale->Create();
    int w = this->ConductanceScale->GetScale()->GetWidth ( );
    this->ConductanceScale->SetRange(0,10);
    this->ConductanceScale->SetResolution (0.1);
    this->ConductanceScale->SetValue(1.0);
  
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->ConductanceScale->GetWidgetName());

    this->TimeStepScale->SetParent( moduleFrame->GetFrame() );
    this->TimeStepScale->SetLabelText("Time Step");
    this->TimeStepScale->Create();
    this->TimeStepScale->GetScale()->SetWidth ( w );
    this->TimeStepScale->SetRange(0.0, 1.0);
    this->TimeStepScale->SetValue(0.1);
    this->TimeStepScale->SetResolution (0.01);
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->TimeStepScale->GetWidgetName());

    this->NumberOfIterationsScale->SetParent( moduleFrame->GetFrame() );
    this->NumberOfIterationsScale->SetLabelText("Iterations");
    this->NumberOfIterationsScale->Create();
    this->NumberOfIterationsScale->GetScale()->SetWidth ( w );
    this->NumberOfIterationsScale->SetValue(1);
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->NumberOfIterationsScale->GetWidgetName());

    this->FixedVolumeSelector->SetNodeClass("vtkMRMLScalarVolumeNode", 
	NULL, NULL, NULL);
    this->FixedVolumeSelector->SetParent( moduleFrame->GetFrame() );
    this->FixedVolumeSelector->Create();
    this->FixedVolumeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
    this->FixedVolumeSelector->UpdateMenu();

    this->FixedVolumeSelector->SetBorderWidth(2);
    this->FixedVolumeSelector->SetLabelText( "Fixed Volume: ");
    this->FixedVolumeSelector->SetBalloonHelpString(
	"select an input volume from the current mrml scene.");
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->FixedVolumeSelector->GetWidgetName());

    this->MovingVolumeSelector->SetNodeClass("vtkMRMLScalarVolumeNode", 
	NULL, NULL, NULL);
    this->MovingVolumeSelector->SetParent( moduleFrame->GetFrame() );
    this->MovingVolumeSelector->Create();
    this->MovingVolumeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
    this->MovingVolumeSelector->UpdateMenu();

    this->MovingVolumeSelector->SetBorderWidth(2);
    this->MovingVolumeSelector->SetLabelText( "Moving Volume: ");
    this->MovingVolumeSelector->SetBalloonHelpString(
	"select an input volume from the current mrml scene.");
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->MovingVolumeSelector->GetWidgetName());

    this->OutVolumeSelector->SetNodeClass(
	"vtkMRMLScalarVolumeNode", NULL, NULL, "GADVolumeOut");
    this->OutVolumeSelector->SetNewNodeEnabled(1);
    this->OutVolumeSelector->SetParent( moduleFrame->GetFrame() );
    this->OutVolumeSelector->Create();
    this->OutVolumeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
    this->OutVolumeSelector->UpdateMenu();

    this->OutVolumeSelector->SetBorderWidth(2);
    this->OutVolumeSelector->SetLabelText( "Output Volume: ");
    this->OutVolumeSelector->SetBalloonHelpString(
	"select an output volume from the current mrml scene.");
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
	this->OutVolumeSelector->GetWidgetName());

    this->CostFunctionButtonSet->SetParent (moduleFrame->GetFrame());
    this->CostFunctionButtonSet->Create();
    this->CostFunctionButtonSet->SetLabelWidth(8);
    this->CostFunctionButtonSet->SetLabelText("CRC: ");
    this->CostFunctionButtonSet->GetWidget()->PackHorizontallyOn();
    vtkKWRadioButton* bt0 
	= this->CostFunctionButtonSet->GetWidget()->AddWidget(0);
    vtkKWRadioButton* bt1 
	= this->CostFunctionButtonSet->GetWidget()->AddWidget(1);
    bt0->SetText("Check");
    bt1->SetText("Ignore");
    bt0->SelectedStateOn();
    this->Script("pack %s -side left -anchor w -fill x -padx 2 -pady 2", 
	this->CostFunctionButtonSet->GetWidgetName());

    this->ApplyButton->SetParent( moduleFrame->GetFrame() );
    this->ApplyButton->Create();
    this->ApplyButton->SetText("Apply");
    this->ApplyButton->SetWidth ( 8 );
    app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
	this->ApplyButton->GetWidgetName());

    moduleFrame->Delete();
}
