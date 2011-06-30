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
#include <vtkLookupTable.h>
#include <vtkMetaImageWriter.h>

#include <algorithm>
#include <math.h>

// default VTK config:
vtkCxxRevisionMacro(vtk2DSceneImageInteractorStyle, "1.6")
vtkStandardNewMacro(vtk2DSceneImageInteractorStyle)

vtk2DSceneImageInteractorStyle::vtk2DSceneImageInteractorStyle()
  : vtkInteractorStyleImage()
{
  PseudoAltKeyFlag = false;
  Renderer = NULL;
  RenderWindow = NULL;
  ReferenceImage = NULL;
  Magnifier = NULL;
  ImageAxesOrientation = NULL;
  ImageActor = NULL;
  ImageMapper = NULL;
  CurrentMagnification = -10000;
  CurrentResliceOrigin[0] = 0;
  CurrentResliceOrigin[1] = 0;
  CurrentResliceOrigin[2] = 0;
  CurrentResliceSpacing[0] = 0;
  CurrentResliceSpacing[1] = 0;
  CurrentResliceSpacing[2] = 0;
  CurrentResliceExtent[0] = 0;
  CurrentResliceExtent[1] = 0;
  CurrentResliceExtent[2] = 0;
  CurrentResliceExtent[3] = 0;
  CurrentResliceExtent[4] = 0;
  CurrentResliceExtent[5] = 0;
  CurrentResliceCenter[0] = 0;
  CurrentResliceCenter[1] = 0;
  CurrentResliceCenter[2] = 0;
  ContinuousZoomCenter[0] = 0;
  ContinuousZoomCenter[1] = 0;
  SupportColorWindowing = true;
  InitialWL[0] = 0;
  InitialWL[1] = 0;
  WLStartPosition[0] = 0;
  WLStartPosition[1] = 0;
  CurrentWLFactor = 1.;
  CurrentWindowLevelChannel = 0; // relate window/level to image mapper!
  MainWindowLevelChannel = NULL;
  MainWindowLevelChannelResetWindowLevel[0] = 255;
  MainWindowLevelChannelResetWindowLevel[1] = 127.5;
  WindowLevelMouseSensitivity = 1.0; // default
  RealTimeMouseSensitivityAdaption = true; // default
  MinimumSpacingForZoom = -1;
  MaximumSpacingForZoom = -1;
  UseMinimumMaximumSpacingForZoom = false;

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
  ImageAxesOrientation = NULL;
  ImageActor = NULL;
  ImageMapper = NULL;
}

void vtk2DSceneImageInteractorStyle::SetMagnifier(vtkImageReslice *magnifier)
{
	if (magnifier != this->Magnifier)
	{
		this->Magnifier = magnifier;
		if (this->ImageAxesOrientation && this->Magnifier)
			this->Magnifier->SetResliceAxes(this->ImageAxesOrientation);
		this->Modified();
	}
}

void vtk2DSceneImageInteractorStyle::SetImageAxesOrientation(vtkMatrix4x4 *orientation)
{
	if (orientation != this->ImageAxesOrientation)
	{
		this->ImageAxesOrientation = orientation;
		if (this->Magnifier)
			this->Magnifier->SetResliceAxes(this->ImageAxesOrientation);
		this->Modified();
	}
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

void vtk2DSceneImageInteractorStyle::GetAdjustedReferenceImageGeometry(
    double spacing[3], int extent[6])
{
  if (ReferenceImage)
  {
    ReferenceImage->GetSpacing(spacing);
    ReferenceImage->GetWholeExtent(extent);
    if (spacing[0] != spacing[1])
    {
      double f, dist;
      if (spacing[0] < spacing[1]) // adjust vertically
      {
        dist = static_cast<double>(extent[3] - extent[2] + 1) * spacing[1];
        f = spacing[1] / spacing[0]; // magnific.
        spacing[1] = spacing[0]; // new spacing!
        extent[3] = static_cast<int>(extent[2] +
            ceil(dist / spacing[1]) - 1); // new extent!
      }
      else // adjust horizontally
      {
//        f = spacing[1] / spacing[0]; // magnific.
//        spacing[1] = spacing[0]; // take over
      }
    }
  }
}

void vtk2DSceneImageInteractorStyle::FitImageToRenderWindow()
{
  if (RenderWindow && ReferenceImage)
  {
    // -> dimensions are derived from the whole extent which is reliable:
    int dims[3];
    int we[6];
    double spacing[3];
    this->GetAdjustedReferenceImageGeometry(spacing, we);

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
    {
      if (imgaspect >= 1)
      {
        if (imgaspect <= vpaspect)
        {
          fitToHeight = true;
        }
        else
        {
          fitToHeight = false;
        }
      }
      else
      {
        fitToHeight = true;
      }
    }
    else // vpaspect < 1
    {
      if (imgaspect >= 1)
      {
        fitToHeight = false;
      }
      else if (imgaspect <= vpaspect)
      {
        fitToHeight = true;
      }
      else
      {
        fitToHeight = false;
      }
    }
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
    spacing[0] /= CurrentMagnification;
    spacing[1] /= CurrentMagnification;
    spacing[2] = 1; // not considered
    int extent[6];
    extent[0] = 0;
    extent[1] = sz[0] - 1;
    extent[2] = 0;
    extent[3] = sz[1] - 1;
    extent[4] = 0; // not considered
    extent[5] = 0;
    // -> correct the image origin:
    double origin[3];
    ReferenceImage->GetOrigin(origin);
    double v[3];
    double offset;
    if (fitToHeight) // -> apply offset along row-direction if necessary
    {
      v[0] = ImageAxesOrientation->Element[0][0]; // image row direction
      v[1] = ImageAxesOrientation->Element[0][1];
      v[2] = ImageAxesOrientation->Element[0][2];
      offset = ((planeWidth - w) / 2) * spacing[0];
    }
    else // -> apply offset along column-direction if necessary
    {
      v[0] = ImageAxesOrientation->Element[1][0]; // image column direction
      v[1] = ImageAxesOrientation->Element[1][1];
      v[2] = ImageAxesOrientation->Element[1][2];
      offset = ((planeHeight - h) / 2) * spacing[1];
    }
    origin[0] += v[0] * offset;
    origin[1] += v[1] * offset;
    origin[2] += v[2] * offset;

    ApplyCurrentGeometry(origin, spacing, extent);
  }
}

void vtk2DSceneImageInteractorStyle::ApplyCurrentGeometry(double *origin,
    double *spacing, int *extent)
{
  if (Magnifier && ImageActor && RenderWindow)
  {
    // image actor is always at postion 0,0; ensure that!
    double *actorPos = ImageActor->GetPosition();
    if (actorPos[0] != 0 || actorPos[1] != 0)
    {
      double zero[2] = {0, 0};
      ImageActor->SetPosition(zero);
    }
    CurrentResliceOrigin[0] = origin[0];
    CurrentResliceOrigin[1] = origin[1];
    CurrentResliceOrigin[2] = origin[2];
  	CurrentResliceSpacing[0] = spacing[0];
  	CurrentResliceSpacing[1] = spacing[1];
  	CurrentResliceSpacing[2] = spacing[2];
  	CurrentResliceExtent[0] = extent[0];
    CurrentResliceExtent[1] = extent[1];
    CurrentResliceExtent[2] = extent[2];
    CurrentResliceExtent[3] = extent[3];
    CurrentResliceExtent[4] = extent[4];
    CurrentResliceExtent[5] = extent[5];

    // compute current center point of resliced image:
    double v1[3];
    v1[0] = ImageAxesOrientation->Element[0][0];
    v1[1] = ImageAxesOrientation->Element[0][1];
    v1[2] = ImageAxesOrientation->Element[0][2];
    double v2[3];
    v2[0] = ImageAxesOrientation->Element[1][0];
    v2[1] = ImageAxesOrientation->Element[1][1];
    v2[2] = ImageAxesOrientation->Element[1][2];
    double currWh = CurrentResliceSpacing[0] * static_cast<double>(
        CurrentResliceExtent[1] - CurrentResliceExtent[0] + 1) / 2.;
    double currHh = CurrentResliceSpacing[1] * static_cast<double>(
        CurrentResliceExtent[3] - CurrentResliceExtent[2] + 1) / 2.;
    CurrentResliceCenter[0] = CurrentResliceOrigin[0] +
        v1[0] * currWh + v2[0] * currHh;
    CurrentResliceCenter[1] = CurrentResliceOrigin[1] +
        v1[1] * currWh + v2[1] * currHh;
    CurrentResliceCenter[2] = CurrentResliceOrigin[2] +
        v1[2] * currWh + v2[2] * currHh;

    Magnifier->SetOutputOrigin(CurrentResliceOrigin);
    Magnifier->SetOutputSpacing(CurrentResliceSpacing);
    Magnifier->SetOutputExtent(CurrentResliceExtent);

  	// checks in order to prevent pipeline errors:
  	Magnifier->GetOutput()->UpdateInformation();
    int we[6];
    Magnifier->GetOutput()->GetWholeExtent(we);
    int ue[6];
    Magnifier->GetOutput()->GetUpdateExtent(ue);
    // (forget about 3rd dimension here - is 2D only)
    if (ue[0] < we[0] || ue[0] > we[1] || ue[1] < ue[0] || ue[1] > we[1] ||
        ue[2] < we[2] || ue[2] > we[3] || ue[3] < ue[2] || ue[3] > we[3])
    {
      Magnifier->GetOutput()->SetUpdateExtentToWholeExtent();
    }

  	Magnifier->Update();
  }
}

void vtk2DSceneImageInteractorStyle::RestoreViewSettings()
{
  if (Magnifier && CurrentMagnification != -10000 && RenderWindow)
  {
    int *sz = RenderWindow->GetSize();
    double newWidth = static_cast<double>(sz[0]);
    double oldWidth = static_cast<double>(CurrentResliceExtent[1] -
        CurrentResliceExtent[0] + 1);
    double f = newWidth / oldWidth;

    Zoom(f, 0.5, 0.5);
    // if min/max constraints for zooming are set, these have to be verified
    // internally by using the appropriate zooming directions (yes, I know, this
    // will introduce a minimal error ...)
    if (UseMinimumMaximumSpacingForZoom)
    {
      if (f < 1.)
        Zoom(1.00001, 0.5, 0.5);
      else
        Zoom(0.9999, 0.5, 0.5);
    }
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
    double newMagnification = CurrentMagnification * factor;
    double refSpacing[3];
    double newSpacing[3];
    int we[6];
    this->GetAdjustedReferenceImageGeometry(refSpacing, we);
    // -> new spacing due to magnification
    newSpacing[0] = refSpacing[0] / newMagnification;
    newSpacing[1] = refSpacing[1] / newMagnification;
    newSpacing[2] = 1; // not considered
    if (UseMinimumMaximumSpacingForZoom && MinimumSpacingForZoom > 0 &&
        factor > 1.) // max zoom
    {
      if (newSpacing[0] < MinimumSpacingForZoom ||
          newSpacing[1] < MinimumSpacingForZoom)
      {
        if (newSpacing[0] < newSpacing[1])
          newMagnification = refSpacing[0] / MinimumSpacingForZoom;
        else
          newMagnification = refSpacing[1] / MinimumSpacingForZoom;
        if (fabs(CurrentMagnification - newMagnification) < 1e-6)
          return; // no change - already at maximum zoom -> exit
        newSpacing[0] = refSpacing[0] / newMagnification;
        newSpacing[1] = refSpacing[1] / newMagnification;
        // update factor as well:
        factor = newMagnification / CurrentMagnification;
      }
    }
    if (UseMinimumMaximumSpacingForZoom && MaximumSpacingForZoom > 0 &&
        factor < 1.) // min zoom
    {
      if (newSpacing[0] > MaximumSpacingForZoom ||
          newSpacing[1] > MaximumSpacingForZoom)
      {
        if (newSpacing[0] > newSpacing[1])
          newMagnification = refSpacing[0] / MaximumSpacingForZoom;
        else
          newMagnification = refSpacing[1] / MaximumSpacingForZoom;
        if (fabs(CurrentMagnification - newMagnification) < 1e-6)
          return; // no change - already at maximum zoom -> exit
        newSpacing[0] = refSpacing[0] / newMagnification;
        newSpacing[1] = refSpacing[1] / newMagnification;
        // update factor as well:
        factor = newMagnification / CurrentMagnification;
      }
    }

    int *sz = RenderWindow->GetSize();
    double w = (double) sz[0];
    double h = (double) sz[1];
    double v1[3];
    v1[0] = ImageAxesOrientation->Element[0][0]; // image row direction
    v1[1] = ImageAxesOrientation->Element[0][1];
    v1[2] = ImageAxesOrientation->Element[0][2];
    double v2[3];
    v2[0] = ImageAxesOrientation->Element[1][0]; // image column direction
    v2[1] = ImageAxesOrientation->Element[1][1];
    v2[2] = ImageAxesOrientation->Element[1][2];

    CurrentMagnification = newMagnification; // apply zoom
    int extent[6];
    extent[0] = 0;
    extent[1] = sz[0] - 1;
    extent[2] = 0;
    extent[3] = sz[1] - 1;
    extent[4] = 0; // not considered
    extent[5] = 0;
    // -> correct the image origin:
    double origin[3];
    origin[0] = CurrentResliceCenter[0] -
        v1[0] * w * newSpacing[0] / 2. -
        v2[0] * h * newSpacing[1] / 2.;
    origin[1] = CurrentResliceCenter[1] -
        v1[1] * w * newSpacing[0] / 2. -
        v2[1] * h * newSpacing[1] / 2.;
    origin[2] = CurrentResliceCenter[2] -
        v1[2] * w * newSpacing[0] / 2. -
        v2[2] * h * newSpacing[1] / 2.;

    if (cx != 0.5 || cy != 0.5)
    {
      double refPt[3];
      refPt[0] = CurrentResliceOrigin[0] +
          v1[0] * w * CurrentResliceSpacing[0] * cx +
          v2[0] * h * CurrentResliceSpacing[1] * cy;
      refPt[1] = CurrentResliceOrigin[1] +
          v1[1] * w * CurrentResliceSpacing[0] * cx +
          v2[1] * h * CurrentResliceSpacing[1] * cy;
      refPt[2] = CurrentResliceOrigin[2] +
          v1[2] * w * CurrentResliceSpacing[0] * cx +
          v2[2] * h * CurrentResliceSpacing[1] * cy;

      double dx = vtkMath::Dot(refPt, v1) - vtkMath::Dot(CurrentResliceCenter, v1);
      double dy = vtkMath::Dot(refPt, v2) - vtkMath::Dot(CurrentResliceCenter, v2);
      double dx2 = dx * factor;
      double dy2 = dy * factor;

      origin[0] = origin[0] + v1[0] * (dx2 - dx) + v2[0] * (dy2 - dy);
      origin[1] = origin[1] + v1[1] * (dx2 - dx) + v2[1] * (dy2 - dy);
      origin[2] = origin[2] + v1[2] * (dx2 - dx) + v2[2] * (dy2 - dy);
    }

    ApplyCurrentGeometry(origin, newSpacing, extent);

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
    double v1[3];
    v1[0] = ImageAxesOrientation->Element[0][0]; // image row direction
    v1[1] = ImageAxesOrientation->Element[0][1];
    v1[2] = ImageAxesOrientation->Element[0][2];
    double v2[3];
    v2[0] = ImageAxesOrientation->Element[1][0]; // image column direction
    v2[1] = ImageAxesOrientation->Element[1][1];
    v2[2] = ImageAxesOrientation->Element[1][2];
    double origin[3];
    double ddx = dx * CurrentResliceSpacing[0];
    double ddy = dy * CurrentResliceSpacing[1];
    origin[0] = CurrentResliceOrigin[0] - v1[0] * ddx - v2[0] * ddy;
    origin[1] = CurrentResliceOrigin[1] - v1[1] * ddx - v2[1] * ddy;
    origin[2] = CurrentResliceOrigin[2] - v1[2] * ddx - v2[2] * ddy;
    ApplyCurrentGeometry(origin, CurrentResliceSpacing, CurrentResliceExtent);

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
    if (!MainWindowLevelChannel)
    {
      ImageMapper->SetColorWindow(CurrentWL[0]);
      ImageMapper->SetColorLevel(CurrentWL[1]);
    }
    else // LUT set
    {
      // assume vtkLookupTable or subclass:
      vtkLookupTable *lut = reinterpret_cast<vtkLookupTable *>(
          MainWindowLevelChannel->GetLookupTable());
      double r[2];
      r[0] = CurrentWL[1] - CurrentWL[0] / 2.; // min
      r[1] = CurrentWL[1] + CurrentWL[0] / 2.; // max
      lut->SetTableRange(r);
      lut->Build();
    }
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
    if (!MainWindowLevelChannel)
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
    else // LUT set
    {
      vtkLookupTable *lut = reinterpret_cast<vtkLookupTable *>(
          MainWindowLevelChannel->GetLookupTable());
      double r[2];
      lut->GetTableRange(r);
      InitialWL[0] = r[1] - r[0]; // window
      InitialWL[1] = r[0] + InitialWL[0] / 2.; // level
      // for reliable scalar range during window/level:
      MainWindowLevelChannel->GetInput()->UpdateInformation();
      MainWindowLevelChannel->GetInput()->SetUpdateExtent
        (MainWindowLevelChannel->GetInput()->GetWholeExtent());
      MainWindowLevelChannel->GetInput()->Update();
      input = reinterpret_cast<vtkImageData*>(MainWindowLevelChannel->GetInput());
    }
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
    if (!MainWindowLevelChannel)
    {
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
    else // LUT set
    {
      vtkLookupTable *lut = reinterpret_cast<vtkLookupTable *>(
          MainWindowLevelChannel->GetLookupTable());
      double r[2];
      r[0] = MainWindowLevelChannelResetWindowLevel[1] -
          MainWindowLevelChannelResetWindowLevel[0] / 2.; // min
      r[1] = MainWindowLevelChannelResetWindowLevel[1] +
          MainWindowLevelChannelResetWindowLevel[0] / 2.; // max
      lut->SetTableRange(r);
      lut->Build();
    }
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
