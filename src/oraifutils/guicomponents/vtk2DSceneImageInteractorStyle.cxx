//
#include "vtk2DSceneImageInteractorStyle.h"

#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>
#include <vtkMath.h>
#include <vtkCallbackCommand.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkImageMapToColors.h>
#include <vtkLookupTable.h>

#include <algorithm>

// default VTK config:
vtkCxxRevisionMacro(vtk2DSceneImageInteractorStyle, "1.4")
vtkStandardNewMacro(vtk2DSceneImageInteractorStyle)

vtk2DSceneImageInteractorStyle::vtk2DSceneImageInteractorStyle()
  : vtkInteractorStyleImage()
{
  PseudoAltKeyFlag = false;
  Renderer = NULL;
  RenderWindow = NULL;
  ReferenceImage = NULL;
  Magnifier = NULL;
  ImageActor = NULL;
  ImageMapper = NULL;
  CurrentMagnification = -10000;
  CurrentCenter[0] = 0;
  CurrentCenter[1] = 0;
  ContinuousZoomCenter[0] = 0;
  ContinuousZoomCenter[1] = 0;
  SupportColorWindowing = true;
  InitialWL[0] = 0;
  InitialWL[1] = 0;
  WLStartPosition[0] = 0;
  WLStartPosition[1] = 0;
  CurrentWLFactor = 1.;
  CurrentWindowLevelChannel = 0; // relate window/level to image mapper!
  WindowLevelMouseSensitivity = 1.0; // default
  RealTimeMouseSensitivityAdaption = true; // default

  vtkSmartPointer<vtkCallbackCommand> cbc =
      vtkSmartPointer<vtkCallbackCommand>::New();
  cbc->SetCallback(WindowLevelCallback);
  cbc->SetClientData(this);
  this->AddObserver(vtkCommand::WindowLevelEvent, cbc);
  this->AddObserver(vtkCommand::StartWindowLevelEvent, cbc);
  this->AddObserver(vtkCommand::ResetWindowLevelEvent, cbc);
}

void vtk2DSceneImageInteractorStyle::WindowLevelCallback(vtkObject *caller,
    unsigned long eid, void *clientdata, void *calldata)
{
  vtk2DSceneImageInteractorStyle *thiss = reinterpret_cast<
      vtk2DSceneImageInteractorStyle *>(clientdata);
  if (thiss)
  {
    if (eid == vtkCommand::WindowLevelEvent)
    {
      thiss->WindowLevelInternal();
    }
    else if (eid == vtkCommand::StartWindowLevelEvent)
    {
      thiss->StartWindowLevelInternal();
    }
    else if (eid == vtkCommand::ResetWindowLevelEvent)
    {
      thiss->ResetWindowLevelInternal();
    }
  }
}

vtk2DSceneImageInteractorStyle::~vtk2DSceneImageInteractorStyle()
{
  RemoveAllWindowLevelChannels();
  Renderer = NULL;
  ReferenceImage = NULL;
  Magnifier = NULL;
  ImageActor = NULL;
  ImageMapper = NULL;
}

void vtk2DSceneImageInteractorStyle::SetRenderer(vtkRenderer *ren)
{
  if (ren != Renderer)
  {
    Renderer = ren;
    if (ren)
      RenderWindow = ren->GetRenderWindow();
    else
      RenderWindow = NULL;
    this->Modified();
  }
}

bool vtk2DSceneImageInteractorStyle::GetCurrentPixelSpacing(double spacing[2])
{
  if (ReferenceImage && CurrentMagnification != -10000)
  {
    double s[3];
    ReferenceImage->GetSpacing(s);
    spacing[0] = s[0] / CurrentMagnification;
    spacing[1] = s[1] / CurrentMagnification;
    return true;
  }
  else
  {
    return false;
  }
}

bool vtk2DSceneImageInteractorStyle::GetCurrentImageOffset(double offset[2])
{
  if (ImageActor)
  {
    offset[0] = ImageActor->GetPosition()[0];
    offset[1] = ImageActor->GetPosition()[1];
    return true;
  }
  else
  {
    return false;
  }
}

void vtk2DSceneImageInteractorStyle::FitImageToRenderWindow()
{
  if (RenderWindow && ReferenceImage)
  {
    // -> dimensions are derived from the whole extent which is reliable:
    int dims[3];
    int we[6];
    ReferenceImage->GetWholeExtent(we);
    dims[0] = we[1] - we[0] + 1;
    dims[1] = we[3] - we[2] + 1;
    dims[2] = we[5] - we[4] + 1;
    // NOTE: We do not need to consider the image spacing because this scene
    // is purely pixel-based!
    double width = (double) dims[0];
    double height = (double) dims[1];
    if (height <= 0)
      height = 1.0;
    if (width < 0)
      width = 1.0;
    int *sz = RenderWindow->GetSize();
    double w = (double) sz[0];
    double h = (double) sz[1];
    double vpaspect = w / h; // viewport
    double imgaspect = width / height; // image
    bool fitToHeight = false;
    // we need some logic ...
    if (vpaspect >= 1)
      if (imgaspect >= 1)
        if (imgaspect <= vpaspect)
          fitToHeight = true;
        else
          fitToHeight = false;
      else
        fitToHeight = true;
    else // vpaspect < 1
    if (imgaspect >= 1)
      fitToHeight = false;
    else if (imgaspect <= vpaspect)
      fitToHeight = true;
    else
      fitToHeight = false;
    // calculate resulting plane height and width
    double planeHeight, planeWidth;
    CurrentMagnification = 1.; // common magnification - keep aspect ratio!
    if (fitToHeight)
    {
      CurrentMagnification = h / height;
      planeHeight = h;
      planeWidth = imgaspect * planeHeight;
    }
    else
    {
      CurrentMagnification = w / width;
      planeWidth = w;
      planeHeight = planeWidth / imgaspect;
    }
    double pos[2];
    pos[0] = (w - planeWidth) / 2;
    pos[1] = (h - planeHeight) / 2;
    CurrentCenter[0] = (pos[0] + planeWidth / 2.) / w;
    CurrentCenter[1] = (pos[1] + planeHeight / 2.) / h;
    ApplyMagnificationAndPosition(pos, true);
  }
}

void vtk2DSceneImageInteractorStyle::ApplyMagnificationAndPosition(
    double *position, bool forceUpdateToWholeExtent)
{
  if (Magnifier && ImageActor && RenderWindow)
  {
    // in order to guarantee a fast zooming, we set the update extent to
    // the whole extent only if really necessary (if the current update
    // extent is outside the whole extent); otherwise we simply apply the
    // current update extent which only costs minimal computation time:
    Magnifier->SetAxisMagnificationFactor(0, CurrentMagnification);
    Magnifier->SetAxisMagnificationFactor(1, CurrentMagnification);
    Magnifier->GetOutput()->UpdateInformation();
    int we[6];
    Magnifier->GetOutput()->GetWholeExtent(we);
    int ue[6];
    Magnifier->GetOutput()->GetUpdateExtent(ue);
    bool ueowe = false; // update-extent-outside-whole-extent-flag
    // (forget about 3rd dimension here - is 2D only)
    if (ue[0] < we[0] || ue[0] > we[1] || ue[1] < ue[0] || ue[1] > we[1] ||
        ue[2] < we[2] || ue[2] > we[3] || ue[3] < ue[2] || ue[3] > we[3])
      ueowe = true;
    if (forceUpdateToWholeExtent || ueowe) // need whole extent as update ext.
      Magnifier->GetOutput()->SetUpdateExtentToWholeExtent();
    Magnifier->Update();
    ImageActor->SetPosition(position);
  }
}

void vtk2DSceneImageInteractorStyle::RestoreViewSettings()
{
  if (CurrentMagnification != -10000 && ReferenceImage && RenderWindow)
  {
    // TODO: to be optimized! (currently we do not adapt current magnification!)
    int *sz = RenderWindow->GetSize();
    double w = (double) sz[0];
    double h = (double) sz[1];
    // -> dimensions are derived from the whole extent which is reliable:
    int dims[3];
    int we[6];
    ReferenceImage->GetWholeExtent(we);
    dims[0] = we[1] - we[0] + 1;
    dims[1] = we[3] - we[2] + 1;
    dims[2] = we[5] - we[4] + 1;
    // NOTE: We do not need to consider the image spacing because this scene
    // is purely pixel-based!
    double width = (double) dims[0];
    double height = (double) dims[1];
    width *= CurrentMagnification;
    height *= CurrentMagnification;
    double pos[2];
    pos[0] = CurrentCenter[0] * w - width / 2.;
    pos[1] = CurrentCenter[1] * h - height / 2.;
    ApplyMagnificationAndPosition(pos);
  }
}

void vtk2DSceneImageInteractorStyle::OnKeyDown()
{
  if (this->Interactor->GetShiftKey() ||
      this->Interactor->GetControlKey())
    return;
  if (this->Interactor->GetAltKey()) // unfortunately never true!
  {
    PseudoAltKeyFlag = true;
    return;
  }
  char key = this->Interactor->GetKeyCode();
  if (key == 0) // it's likely that ALT was pressed
    PseudoAltKeyFlag = true;
}

void vtk2DSceneImageInteractorStyle::OnKeyUp()
{
  PseudoAltKeyFlag = false;
}

void vtk2DSceneImageInteractorStyle::OnLeftButtonDown()
{
  if (PseudoAltKeyFlag) // PAN
  {
    this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
                            this->Interactor->GetEventPosition()[1]);
    if (this->CurrentRenderer == NULL)
    {
      return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartPan();
  }
}

void vtk2DSceneImageInteractorStyle::OnLeftButtonUp()
{
  switch (this->State)
  {
    case VTKIS_PAN:
      this->EndPan();
      if (this->Interactor)
      {
        this->ReleaseFocus();
      }
      break;
  }
}

void vtk2DSceneImageInteractorStyle::OnChar()
{
  vtkRenderWindowInteractor *rwi = this->Interactor;

  switch (rwi->GetKeyCode())
  {
    case 'r':
    case 'R':
      if (!rwi->GetControlKey())
      {
        FitImageToRenderWindow();
        rwi->Render();
      }
      break;
  }

  // do not forward, deactivate the other keys
}

void vtk2DSceneImageInteractorStyle::OnKeyPress()
{
  vtkRenderWindowInteractor *rwi = this->Interactor;
  std::string key = rwi->GetKeySym();

  if (key.compare("r") == 0 && rwi->GetControlKey())
  {
    this->InvokeEvent(vtkCommand::ResetWindowLevelEvent);
    return;
  }

  if (key.compare("plus") == 0)
  {
    OnMouseWheelForward(); // zoom-in
    return; // no forwarding
  }
  else if (key.compare("minus") == 0)
  {
    OnMouseWheelBackward(); // zoom-out
    return; // no forwarding
  }

  // forward events
  vtkInteractorStyleImage::OnKeyPress();
}

void vtk2DSceneImageInteractorStyle::Zoom(double factor, double cx, double cy)
{
  if (CurrentMagnification != -10000 && ReferenceImage && RenderWindow)
  {
    CurrentMagnification *= factor;
    int *sz = RenderWindow->GetSize();
    double w = (double) sz[0];
    double h = (double) sz[1];
    // -> dimensions are derived from the whole extent which is reliable:
    int dims[3];
    int we[6];
    ReferenceImage->GetWholeExtent(we);
    dims[0] = we[1] - we[0] + 1;
    dims[1] = we[3] - we[2] + 1;
    dims[2] = we[5] - we[4] + 1;
    // NOTE: We do not need to consider the image spacing because this scene
    // is purely pixel-based!
    double width = (double) dims[0];
    double height = (double) dims[1];
    width *= CurrentMagnification;
    height *= CurrentMagnification;
    double pos[2];
    // correct the position w.r.t. to zoom center:
    double off[2];
    off[0] = (cx - CurrentCenter[0]) * w + (width / factor) / 2.;
    off[1] = (cy - CurrentCenter[1]) * h + (height / factor) / 2.;
    pos[0] = cx * w - off[0] * factor;
    pos[1] = cy * h - off[1] * factor;
    // new center:
    CurrentCenter[0] = (pos[0] + width / 2.) / w;
    CurrentCenter[1] = (pos[1] + height / 2.) / h;
    ApplyMagnificationAndPosition(pos);
    this->Interactor->Render();
  }
}

void vtk2DSceneImageInteractorStyle::OnMouseWheelForward()
{
  this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
      this->Interactor->GetEventPosition()[1]);
  if (this->CurrentRenderer == NULL)
  {
    return;
  }
  double factor = this->MotionFactor * 0.2 * this->MouseWheelMotionFactor;
  factor = pow(1.1, factor);
  this->GrabFocus(this->EventCallbackCommand);
  this->StartZoom();
  Zoom(factor, 0.5, 0.5);
  this->EndZoom();
  this->ReleaseFocus();
  this->InvokeEvent(vtkCommand::InteractionEvent, NULL);
}

void vtk2DSceneImageInteractorStyle::OnMouseWheelBackward()
{
  this->FindPokedRenderer(this->Interactor->GetEventPosition()[0],
      this->Interactor->GetEventPosition()[1]);
  if (this->CurrentRenderer == NULL)
  {
    return;
  }
  double factor = this->MotionFactor * -0.2 * this->MouseWheelMotionFactor;
  factor = pow(1.1, factor);
  this->GrabFocus(this->EventCallbackCommand);
  this->StartZoom();
  Zoom(factor, 0.5, 0.5);
  this->EndZoom();
  this->ReleaseFocus();
  this->InvokeEvent(vtkCommand::InteractionEvent, NULL);
}

void vtk2DSceneImageInteractorStyle::Pan(int dx, int dy)
{
  if (CurrentMagnification != -10000 && ReferenceImage && RenderWindow)
  {
    int *sz = RenderWindow->GetSize();
    double w = (double) sz[0];
    double h = (double) sz[1];
    // -> dimensions are derived from the whole extent which is reliable:
    int dims[3];
    int we[6];
    ReferenceImage->GetWholeExtent(we);
    dims[0] = we[1] - we[0] + 1;
    dims[1] = we[3] - we[2] + 1;
    dims[2] = we[5] - we[4] + 1;
    // NOTE: We do not need to consider the image spacing because this scene
    // is purely pixel-based!
    double width = (double) dims[0];
    double height = (double) dims[1];
    width *= CurrentMagnification;
    height *= CurrentMagnification;
    double pos[2];
    // current position:
    pos[0] = CurrentCenter[0] * w - width / 2.;
    pos[1] = CurrentCenter[1] * h - height / 2.;
    // apply panning:
    pos[0] += dx;
    pos[1] += dy;
    // store new center:
    CurrentCenter[0] = (pos[0] + width / 2.) / w;
    CurrentCenter[1] = (pos[1] + height / 2.) / h;
    ApplyMagnificationAndPosition(pos);
    this->Interactor->Render();
  }
}

void vtk2DSceneImageInteractorStyle::OnMiddleButtonDown()
{
  if (SupportColorWindowing && ImageMapper)
  {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    this->FindPokedRenderer(x, y);
    if (this->CurrentRenderer == NULL)
      return;
    this->GrabFocus(this->EventCallbackCommand);
    this->WindowLevelStartPosition[0] = x;
    this->WindowLevelStartPosition[1] = y;
    this->StartWindowLevel();
  }
}

void vtk2DSceneImageInteractorStyle::OnMiddleButtonUp()
{
  this->EndWindowLevel();
}

void vtk2DSceneImageInteractorStyle::OnRightButtonDown()
{
  if (this->Interactor->GetShiftKey()) // ZOOM - click-point-based
  {
    int *sz = RenderWindow->GetSize();
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];
    ContinuousZoomCenter[0] = (double) x / (double) sz[0];
    ContinuousZoomCenter[1] = (double) y / (double) sz[1];
    this->StartZoom();
  }
  else // ZOOM - center-based
  {
    ContinuousZoomCenter[0] = 0.5;
    ContinuousZoomCenter[1] = 0.5;
    this->StartZoom();
  }
}

void vtk2DSceneImageInteractorStyle::OnRightButtonUp()
{
  this->EndZoom();
}

void vtk2DSceneImageInteractorStyle::OnMouseMove()
{
  switch (this->State)
  {
    case VTKIS_PAN:
    case VTKIS_ZOOM:
      break;
    default:
      this->vtkInteractorStyleImage::OnMouseMove(); // forward
      return; // early exit
  }

  int x = this->Interactor->GetEventPosition()[0];
  int y = this->Interactor->GetEventPosition()[1];
  double *center = NULL;
  int dy = 0;
  double dyf = 0;
  switch (this->State)
  {
    case VTKIS_PAN:
      this->FindPokedRenderer(x, y);
      Pan(x - this->Interactor->GetLastEventPosition()[0], y
          - this->Interactor->GetLastEventPosition()[1]);
      this->InvokeEvent(vtkCommand::InteractionEvent, NULL);
      break;
    case VTKIS_ZOOM:
      this->FindPokedRenderer(x, y);
      center = this->CurrentRenderer->GetCenter();
      dy = this->Interactor->GetEventPosition()[1]
          - this->Interactor->GetLastEventPosition()[1];
      dyf = this->MotionFactor * dy / center[1];
      Zoom(pow(1.1, dyf), ContinuousZoomCenter[0], ContinuousZoomCenter[1]);
      this->InvokeEvent(vtkCommand::InteractionEvent, NULL);
      break;
  }
}

void vtk2DSceneImageInteractorStyle::WindowLevelInternal()
{
  bool ok = SupportColorWindowing;
  if (CurrentWindowLevelChannel < 1 ||
      CurrentWindowLevelChannel > (int)WindowLevelChannels.size())
  {
    if (!ImageMapper || !ImageMapper->GetInput())
      ok = false;
  }
  else
  {
    if (!WindowLevelChannels[CurrentWindowLevelChannel - 1] ||
        !WindowLevelChannels[CurrentWindowLevelChannel - 1]->GetInput())
      ok = false;
  }
  if (!ok)
    return;

  if (RealTimeMouseSensitivityAdaption)
    ComputeCurrentWindowLevelFactors(); // re-compute sensitivity factors

  double fwin = CurrentWLFactor;
  double flev = CurrentWLFactor;
  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];
  if (!RealTimeMouseSensitivityAdaption)
  {
    CurrentWL[0] = InitialWL[0] + fwin * (double)(WLStartPosition[0] - X);
    CurrentWL[1] = InitialWL[1] + flev * (double)(WLStartPosition[1] - Y);
  }
  else
  {
    CurrentWL[0] = CurrentWL[0] + fwin * (double)(WLStartPosition[0] - X);
    CurrentWL[1] = CurrentWL[1] + flev * (double)(WLStartPosition[1] - Y);
    WLStartPosition[0] = X; // update for next window/level step
    WLStartPosition[1] = Y;
  }

  if (CurrentWindowLevelChannel <= 0 ||
      CurrentWindowLevelChannel > (int)WindowLevelChannels.size()) // main chan.
  {
    // finally take over and render
    ImageMapper->SetColorWindow(CurrentWL[0]);
    ImageMapper->SetColorLevel(CurrentWL[1]);
  }
  else
  {
    vtkImageMapToColors *map = WindowLevelChannels[CurrentWindowLevelChannel - 1];
    // assume vtkLookupTable or subclass:
    vtkLookupTable *lut = reinterpret_cast<vtkLookupTable *>(map->GetLookupTable());
    double r[2];
    r[0] = CurrentWL[1] - CurrentWL[0] / 2.; // min
    r[1] = CurrentWL[1] + CurrentWL[0] / 2.; // max
    lut->SetTableRange(r);
    lut->Build();
  }

  this->Interactor->Render();
}

void vtk2DSceneImageInteractorStyle::StartWindowLevelInternal()
{
  bool ok = SupportColorWindowing;
  if (CurrentWindowLevelChannel < 1 ||
      CurrentWindowLevelChannel > (int)WindowLevelChannels.size())
  {
    if (!ImageMapper || !ImageMapper->GetInput())
      ok = false;
  }
  else
  {
    if (!WindowLevelChannels[CurrentWindowLevelChannel - 1] ||
        !WindowLevelChannels[CurrentWindowLevelChannel - 1]->GetInput())
      ok = false;
  }
  if (!ok)
    return;

  WLStartPosition[0] = this->Interactor->GetEventPosition()[0];
  WLStartPosition[1] = this->Interactor->GetEventPosition()[1];

  vtkImageData *input = NULL;
  if (CurrentWindowLevelChannel <= 0 ||
      CurrentWindowLevelChannel > (int)WindowLevelChannels.size()) // main chan.
  {
    InitialWL[0] = ImageMapper->GetColorWindow(); // store initial situation
    InitialWL[1] = ImageMapper->GetColorLevel();
    // for reliable scalar range during window/level:
    ImageMapper->GetInput()->UpdateInformation();
    ImageMapper->GetInput()->SetUpdateExtent
      (ImageMapper->GetInput()->GetWholeExtent());
    ImageMapper->GetInput()->Update();
    input = ImageMapper->GetInput();
  }
  else // further window/level channel
  {
    vtkImageMapToColors *map = WindowLevelChannels[CurrentWindowLevelChannel - 1];
    // assume vtkLookupTable or subclass:
    vtkLookupTable *lut = reinterpret_cast<vtkLookupTable *>(map->GetLookupTable());
    double r[2];
    lut->GetTableRange(r);
    InitialWL[0] = r[1] - r[0]; // window
    InitialWL[1] = r[0] + InitialWL[0] / 2.; // level
    // for reliable scalar range during window/level:
    map->GetInput()->UpdateInformation();
    map->GetInput()->SetUpdateExtent
      (map->GetInput()->GetWholeExtent());
    map->GetInput()->Update();
    input = reinterpret_cast<vtkImageData*>(map->GetInput());
  }
  CurrentWL[0] = InitialWL[0];
  CurrentWL[1] = InitialWL[1];

  int components = input->GetPointData()->GetScalars()->GetNumberOfComponents();
  if (components == 3)
  {
    double sr0[2];
    double sr1[2];
    double sr2[2];
    input->GetPointData()->GetScalars()->GetRange(sr0, 0);
    input->GetPointData()->GetScalars()->GetRange(sr1, 1);
    input->GetPointData()->GetScalars()->GetRange(sr2, 2);
    CurrentSR[0] = std::min(sr0[0], std::min(sr1[0], sr2[0]));
    CurrentSR[1] = std::max(sr0[1], std::max(sr1[1], sr2[1]));
  }
  else
  {
    input->GetScalarRange(CurrentSR);
  }

  // compute sensitivity factors dependent on current scalar range
  // (CurrentSR), current window/level (CurrentWL) and current
  // mouse sensitivity control (WindowLevelMouseSensitivity):
  ComputeCurrentWindowLevelFactors();
}

void vtk2DSceneImageInteractorStyle::ComputeCurrentWindowLevelFactors()
{
  double contrast = 1 - CurrentWL[0] / (CurrentSR[1] - CurrentSR[0]);
  //double brightness = 1 - (CurrentWL[1] - CurrentSR[0]) / (CurrentSR[1] - CurrentSR[0]);
  double diffc = 1. - contrast;
  if (diffc > 2.)
    diffc = 2.;
  CurrentWLFactor = (exp(diffc) - 1.) / (exp(1.) - 1.) *
      WindowLevelMouseSensitivity + 0.01;
  // factor is optimized for a span width of 255 -> adapt to real image span
  // width:
  CurrentWLFactor = CurrentWLFactor / 255. * (CurrentSR[1] - CurrentSR[0]);
}

void vtk2DSceneImageInteractorStyle::ResetWindowLevelInternal()
{
  bool ok = SupportColorWindowing;
  if (CurrentWindowLevelChannel < 1 ||
      CurrentWindowLevelChannel > (int)WindowLevelChannels.size())
  {
    if (!ImageMapper || !ImageMapper->GetInput())
      ok = false;
  }
  else
  {
    if (!WindowLevelChannels[CurrentWindowLevelChannel - 1] ||
        !WindowLevelChannels[CurrentWindowLevelChannel - 1]->GetInput())
      ok = false;
  }
  if (!ok)
    return;

  this->StartWindowLevel();

  if (CurrentWindowLevelChannel <= 0 ||
      CurrentWindowLevelChannel > (int)WindowLevelChannels.size()) // main chan.
  {
    double sr[2];
    ImageMapper->GetInput()->UpdateInformation();
    ImageMapper->GetInput()->SetUpdateExtent
      (ImageMapper->GetInput()->GetWholeExtent());
    ImageMapper->GetInput()->Update();
    int components = ImageMapper->GetInput()->GetPointData()->GetScalars()->
        GetNumberOfComponents();
    if (components == 3)
    {
      double sr0[2];
      double sr1[2];
      double sr2[2];
      ImageMapper->GetInput()->GetPointData()->GetScalars()->GetRange(sr0, 0);
      ImageMapper->GetInput()->GetPointData()->GetScalars()->GetRange(sr1, 1);
      ImageMapper->GetInput()->GetPointData()->GetScalars()->GetRange(sr2, 2);
      sr[0] = std::min(sr0[0], std::min(sr1[0], sr2[0]));
      sr[1] = std::max(sr0[1], std::max(sr1[1], sr2[1]));
    }
    else
    {
      ImageMapper->GetInput()->GetScalarRange(sr);
    }
    ImageMapper->SetColorWindow(sr[1] - sr[0]);
    ImageMapper->SetColorLevel((sr[1] - sr[0]) / 2.);
  }
  else // further window/level channel
  {
    double *wl = WindowLevelChannelsResetWindowLevels[CurrentWindowLevelChannel - 1];
    vtkImageMapToColors *map = WindowLevelChannels[CurrentWindowLevelChannel - 1];
    // assume vtkLookupTable or subclass:
    vtkLookupTable *lut = reinterpret_cast<vtkLookupTable *>(map->GetLookupTable());
    double r[2];
    r[0] = wl[1] - wl[0] / 2.; // min
    r[1] = wl[1] + wl[0] / 2.; // max
    lut->SetTableRange(r);
    lut->Build();
  }

  this->Interactor->Render();
  this->EndWindowLevel();
}

int vtk2DSceneImageInteractorStyle::AddWindowLevelChannel(
    vtkImageMapToColors *channel, double *resetWindowLevel)
{
  if (!resetWindowLevel)
    return 0;

  WindowLevelChannels.push_back(channel);
  double *wl = new double[2];
  wl[0] = resetWindowLevel[0];
  wl[1] = resetWindowLevel[1];
  WindowLevelChannelsResetWindowLevels.push_back(wl);

  return WindowLevelChannels.size();
}

bool vtk2DSceneImageInteractorStyle::RemoveWindowLevelChannel(
    vtkImageMapToColors *channel)
{
  std::vector<vtkImageMapToColors *>::iterator it;
  int idx = 0;
  for (it = WindowLevelChannels.begin(); it != WindowLevelChannels.end(); ++it)
  {
    idx++;
    if (*it == channel)
    {
      // -> adapt selection if necessary
      if (CurrentWindowLevelChannel >= 1 &&
          CurrentWindowLevelChannel <= (int)WindowLevelChannels.size())
      {
        if (idx == CurrentWindowLevelChannel)
          CurrentWindowLevelChannel = 0; // invalidate
        else if (CurrentWindowLevelChannel > idx)
          CurrentWindowLevelChannel--; // adjust
      }
      WindowLevelChannels.erase(it);
      if (WindowLevelChannelsResetWindowLevels[idx - 1])
        delete[] WindowLevelChannelsResetWindowLevels[idx - 1];
      WindowLevelChannelsResetWindowLevels.erase(
          WindowLevelChannelsResetWindowLevels.begin() + idx - 1);
      return true;
    }
  }
  return false;
}

int vtk2DSceneImageInteractorStyle::GetIndexOfWindowLevelChannel(
    vtkImageMapToColors *channel)
{
  std::vector<vtkImageMapToColors *>::iterator it;
  int idx = 0;
  for (it = WindowLevelChannels.begin(); it != WindowLevelChannels.end(); ++it)
  {
    idx++;
    if (*it == channel)
      return idx;
  }
  return 0;
}

void vtk2DSceneImageInteractorStyle::RemoveAllWindowLevelChannels()
{
  WindowLevelChannels.clear();
  CurrentWindowLevelChannel = 0;
  for (std::size_t i = 0; i < WindowLevelChannelsResetWindowLevels.size(); i++)
  {
    if (WindowLevelChannelsResetWindowLevels[i])
      delete[] WindowLevelChannelsResetWindowLevels[i];
  }
  WindowLevelChannelsResetWindowLevels.clear();
}

bool vtk2DSceneImageInteractorStyle::OverrideResetWindowLevel(int index,
    double *resetWindowLevel)
{
  if (!resetWindowLevel)
    return false;

  if (index >= 1 && index <= (int)WindowLevelChannels.size())
  {
    WindowLevelChannelsResetWindowLevels[index - 1][0] = resetWindowLevel[0];
    WindowLevelChannelsResetWindowLevels[index - 1][1] = resetWindowLevel[1];
    return true;
  }
  return false;
}

bool vtk2DSceneImageInteractorStyle::OverrideResetWindowLevelByMinMax(int index,
    double *resetMinMax)
{
  if (!resetMinMax)
    return false;
  double wl[2];
  wl[0] = resetMinMax[1] - resetMinMax[0]; // window
  wl[1] = resetMinMax[0] + wl[0] / 2.; // level
  return OverrideResetWindowLevel(index, wl);
}
