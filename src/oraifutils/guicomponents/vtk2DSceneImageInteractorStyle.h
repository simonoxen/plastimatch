//
#ifndef VTK2DSCENEIMAGEINTERACTORSTYLE_H_
#define VTK2DSCENEIMAGEINTERACTORSTYLE_H_

#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageData.h>
#include <vtkActor2D.h>
#include <vtkImageMapper.h>
#include <vtkImageResample.h>

#include <vector>

// forward declaration:
class vtkRenderWindow;
class vtkImageMapToColors;

/**
 * Interactor style dedicated for usage with 2D scenes that are based on
 * vtkActor2D and vtkImageMapper. It enables basic zooming, panning, image
 * fitting (adapting image zoom to viewport), and color windowing.
 *
 * The following external component setup is assumed:<br>
 * An input image (or a specified image filter output) is connected to a
 * vtkImageResample instance (the magnifier for zooming capability). The output
 * of the magnifier is connected to an instance of vtkImageMapper which is
 * connected to a vtkActor2D instance itself. The reference to this actor must
 * be set to this interactor style. Moreover, the magnifier instance must be
 * set. In addition, the pointer to the reference image (e.g. the input of
 * vtkImageResample) must be set; this is important for zooming and image
 * fitting. In order to support color windowing, the vtkImageMapper pointer must
 * be set.
 *
 * Moreover, this interactor style supports color windowing of further "channels"
 * different from the main input image mapper. These optional further window/
 * level "channels" can be dynamically added/removed. Depending on the
 * CurrentWindowLevelChannel property, color windowing relates either to the
 * main image mapper or to the specified optional window/level channel.
 *
 * Windowing is related relatively to contrast c=1.0. Near c=1.0 the
 * sensitivity is minimal. In opposite, sensitivity is nonlinearly increasing
 * with increasing deviation from c=1.0. Window and level sensitivities are
 * adapted in parallel; i.e. as soon as the contrast reaches its maximum,
 * leveling is also sensitive.
 *
 * Key/mouse bindings:<br>
 * - ZOOMING: (a) right mouse-button and up/down-movement (zooming w.r.t.
 * viewport center), (b) right mouse-button+shift and up/down-movement (zooming
 * w.r.t. to click-position), (c) '+'/'-'-keys (zoom-in, zoom-out), (d) 'r'-key
 * (fit zoom to actual viewport)<br>
 * - PANNING: left mouse-button + ALT-key and left/right/up/down-movement
 * (panning of the image)<br>
 * - COLOR WINDOWING: (a) middle mouse-button (up-movement: increase
 * brightness, down-movement: decrease brightness, left-movement: decrease
 * contrast, right-movement: increase contrast), (b) 'r'-key+ctrl (reset window
 * level to whole image range) <br>
 *
 * @see vtkInteractorStyleImage
 * @see vtkActor2D
 * @see vtkImageMapper
 * @see vtkImageResample
 *
 * @author phil 
 * @version 1.4
 */
class vtk2DSceneImageInteractorStyle :
  public vtkInteractorStyleImage
{
public:
  /** Standard new **/
  static vtk2DSceneImageInteractorStyle* New();
  vtkTypeRevisionMacro(vtk2DSceneImageInteractorStyle, vtkInteractorStyleImage);

  /** Fit the image to the render window. **/
  void FitImageToRenderWindow();

  void SetRenderer(vtkRenderer *ren);
  vtkGetObjectMacro(Renderer, vtkRenderer)

  vtkSetObjectMacro(ImageActor, vtkActor2D)
  vtkGetObjectMacro(ImageActor, vtkActor2D)

  vtkSetObjectMacro(Magnifier, vtkImageResample)
  vtkGetObjectMacro(Magnifier, vtkImageResample)

  vtkSetObjectMacro(ImageMapper, vtkImageMapper)
  vtkGetObjectMacro(ImageMapper, vtkImageMapper)

  vtkSetObjectMacro(ReferenceImage, vtkImageData)
  vtkGetObjectMacro(ReferenceImage, vtkImageData)

  vtkSetMacro(SupportColorWindowing, bool)
  vtkGetMacro(SupportColorWindowing, bool)
  vtkBooleanMacro(SupportColorWindowing, bool)

  /** Restore the last saved relative positioning scheme (center position,
   * magnification). May be useful when render window size is altered. **/
  void RestoreViewSettings();

  /** Mouse scroll event binding: zoom in. **/
  virtual void OnMouseWheelForward();
  /** Mouse scroll event binding: zoom out. **/
  virtual void OnMouseWheelBackward();
  /** Mouse move bindings: realize the operations here. **/
  virtual void OnMouseMove();
  /** Right mouse button event bindings:
   * (NO MODIFIER): center-based zooming<br>
   * (SHIFT): click-point-based zooming<br>
   * (CONTROL): Window/Level<br>
   **/
  virtual void OnRightButtonDown();
  /** Right mouse button event bindings:
   * (NO MODIFIER): center-based zooming<br>
   * (SHIFT): click-point-based zooming<br>
   * (CONTROL): Window/Level<br>
   **/
  virtual void OnRightButtonUp();
  /** Middle mouse button event bindings:
   * (NO MODIFIER): Window/Level<br>
   **/
  virtual void OnMiddleButtonDown();
  /** Middle mouse button event bindings:
   * (NO MODIFIER): Window/Level<br>
   **/
  virtual void OnMiddleButtonUp();
  /** Left mouse button event bindings:
   * (ALT): panning
   **/
  virtual void OnLeftButtonDown();
  /** Left mouse button event bindings:
   * (ALT): panning
   **/
  virtual void OnLeftButtonUp();

  /** A key goes down (for pseudo-ALT-flag). **/
  virtual void OnKeyDown();
  /** A key goes up (for pseudo-ALT-flag). **/
  virtual void OnKeyUp();

  /** Return the current size of a pixel w.r.t. to current magnification.
   * If the pixel spacing cannot be returned, FALSE is returned - otherwise,
   * if x and y pixel spacing can be returned, TRUE is returned. **/
  bool GetCurrentPixelSpacing(double spacing[2]);
  /** Return the current offset of the image (actor) in mm.
   * If the offset cannot be returned, FALSE is returned - otherwise,
   * if the offset can be returned, TRUE is returned. **/
  bool GetCurrentImageOffset(double offset[2]);

  /** Get the array of optional further LUTs for other window/level "channels" **/
  const std::vector<vtkImageMapToColors *> &GetWindowLevelChannels()
  {
    return WindowLevelChannels;
  }
  /** Add an optional further mapper as other window/level "channel" and return its
   * 1-based index. In addition, reset window and level values must be defined
   * (2-component double array, 1st component is window, 2nd is level).
   * NOTE: We assume the mapper's lookup table being an instance of
   * vtkLookupTable or an instance of a subclass of vtkLookupTable. **/
  int AddWindowLevelChannel(vtkImageMapToColors *channel, double *resetWindowLevel);
  /** Override the reset window/level values of the specified further window/
   * level "channel".
   * @return TRUE if the values were successfully overridden **/
  bool OverrideResetWindowLevel(int index, double *resetWindowLevel);
  /** Override the reset window/level values of the specified further window/
   * level "channel" by specifying minimum and maximum of the intensity range.
   * @return TRUE if the values were successfully overridden **/
  bool OverrideResetWindowLevelByMinMax(int index, double *resetMinMax);
  /** Remove the specified LUT from window/level "channel"s and return TRUE if
   * successful. **/
  bool RemoveWindowLevelChannel(vtkImageMapToColors *channel);
  /** Return the 1-based index of the specified window/level "channel". 0 is
   * returned if the channel is not found in the internal array. */
  int GetIndexOfWindowLevelChannel(vtkImageMapToColors *channel);
  /** Remove all window/level channels. **/
  void RemoveAllWindowLevelChannels();

  vtkSetMacro(CurrentWindowLevelChannel, int)
  vtkGetMacro(CurrentWindowLevelChannel, int)

  vtkSetMacro(WindowLevelMouseSensitivity, double)
  vtkGetMacro(WindowLevelMouseSensitivity, double)

  vtkSetMacro(RealTimeMouseSensitivityAdaption, bool)
  vtkGetMacro(RealTimeMouseSensitivityAdaption, bool)
  vtkBooleanMacro(RealTimeMouseSensitivityAdaption, bool)

protected:
  /** Flag is true if we think that the ALT key is pressed. The ALT key is not
   * really trackable using the interactor style. BUT: if the returned key code
   * is 0 and neither CTRL nor SHIFT are active, it's quite likely that ALT is
   * pressed. **/
  bool PseudoAltKeyFlag;
  /** Connected renderer **/
  vtkRenderer *Renderer;
  /** Reference to the renderer's render window **/
  vtkRenderWindow *RenderWindow;
  /** Connected reference (fixed) image for resetting the camera **/
  vtkImageData *ReferenceImage;
  /** Magnifier that is connected to the image mapper's input and internally
   * realizes zooming **/
  vtkImageResample *Magnifier;
  /** Image actor under investigation (2D actor) **/
  vtkActor2D *ImageActor;
  /** Image mapper under investigation (2D mapper) **/
  vtkImageMapper *ImageMapper;
  /** Holds current magnification factor **/
  double CurrentMagnification;
  /** Holds current relative actor center position. **/
  double CurrentCenter[2];
  /** Stores the zoom center for continuous zooming (right mouse button). **/
  double ContinuousZoomCenter[2];
  /** Flag indicating whether or not to support window/level. Needs a valid
   * pointer to the 2D image mapper.
   * @see ImageMapper **/
  bool SupportColorWindowing;
  /** Helper for color windowing. **/
  double InitialWL[2];
  /** Helper for color windowing. **/
  double CurrentWL[2];
  /** Helper for color windowing. **/
  double CurrentSR[2];
  /** Helper for color windowing. **/
  int WLStartPosition[2];
  /** Current window/level sensitivity factor. **/
  double CurrentWLFactor;
  /** Mouse sensitivity for color windowing: a value <1.0 decreases sensitivity,
   * a value >1.0 increases sensitivity. **/
  double WindowLevelMouseSensitivity;
  /** Flag indicating whether mouse sensitivity for window/level should be
   * adapted in dependence of contrast in real-time during windowing. If TRUE,
   * sensitivity is real-time-adapted, if FALSE, sensitivity is adapted each
   * time the user initiates a new windowing (by pressing the mouse down). **/
  bool RealTimeMouseSensitivityAdaption;
  /** Array of optional further LUTs for other window/level "channels" **/
  std::vector<vtkImageMapToColors *> WindowLevelChannels;
  /** Array corresponding to WindowLevelChannels with optional "reset" window/
   * level values for each channel (NULL if not specified -> reset window/level
   * will then fit to the whole image range!). **/
  std::vector<double *> WindowLevelChannelsResetWindowLevels;
  /** Indicate which window/level "channel" is currently selected. The index
   * starts with 1 and ends with number of window/level "channels". An index
   * of 0 or another invalid index implicitly selects the image mapper
   * window/level "channel"! **/
  int CurrentWindowLevelChannel;

  /** Callback for color windowing interactions. **/
  static void WindowLevelCallback(vtkObject *caller, unsigned long eid,
      void *clientdata, void *calldata);

  /** Default constructor **/
  vtk2DSceneImageInteractorStyle();
  /** Hidden default destructor. **/
  virtual ~vtk2DSceneImageInteractorStyle();

  /** @see vtkInteractorStyle#OnChar() **/
  virtual void OnChar();
  /** @see vtkInteractorStyle#OnKeyPress() **/
  virtual void OnKeyPress();

  /** Zoom in/out by a specified factor. A factor>1.0 zooms in, a factor<1.0
   * zooms out. cx and cy specify the relative position of the zoom center. **/
  virtual void Zoom(double factor, double cx, double cy);

  /** Pan the image by the specified offset (in pixels). **/
  virtual void Pan(int dx, int dy);

  /** Apply current magnification and specified position (left lower corner).**/
  void ApplyMagnificationAndPosition(double *position,
      bool forceUpdateToWholeExtent = false);

  /** Internal implementation of color windowing (ORA style). **/
  virtual void WindowLevelInternal();
  /** Internal implementation of color windowing start. **/
  virtual void StartWindowLevelInternal();
  /** Internal implementation of color windowing reset. **/
  virtual void ResetWindowLevelInternal();

  /** Compute current window/level factors that control w/l-sensitivity. **/
  virtual void ComputeCurrentWindowLevelFactors();

private:
  // purposely not implemented
  vtk2DSceneImageInteractorStyle(const vtk2DSceneImageInteractorStyle &);
  // purposely not implemented
  void operator=(const vtk2DSceneImageInteractorStyle &);

};

#endif /* VTK2DSCENEIMAGEINTERACTORSTYLE_H_ */
