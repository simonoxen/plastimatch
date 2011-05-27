
SET(LIB_NAME UNO23RegGUI)

INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}         # For uno23reginfo.h (oraUNO23AboutDialog.h)
  ${CMAKE_CURRENT_BINARY_DIR} # For headers produced by UIC
)

QT4_ADD_RESOURCES(RCC_SRCS
  oraUNO23ControlWindow.qrc
  oraUNO23RenderViewDialog.qrc
  oraUNO23TaskPresentationWidget.qrc
  oraUNO23AboutDialog.qrc
)

QT4_WRAP_UI(UI_SRCS 
  oraUNO23ControlWindow.ui
  oraUNO23RenderViewDialog.ui
  oraUNO23TaskPresentationWidget.ui
  oraUNO23AboutDialog.ui
)  

QT4_WRAP_CPP(MOC_SRCS 
  oraUNO23ControlWindow.h
  oraUNO23TaskPresentationWidget.h
  oraUNO23RenderViewDialog.h
  oraUNO23AboutDialog.h
)

ADD_LIBRARY(${LIB_NAME}
  ${RCC_SRCS}
  ${UI_SRCS}
  ${MOC_SRCS}
  oraUNO23ControlWindow.cxx
  oraUNO23TaskPresentationWidget.cxx
  oraUNO23RenderViewDialog.cxx 	
  oraUNO23AboutDialog.cxx
  vtkPerspectiveOverlayProjectionInteractorStyle.h
  vtkPerspectiveOverlayProjectionInteractorStyle.cxx 	
)

SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
TARGET_LINK_LIBRARIES(${LIB_NAME} 
  ${QT_QTCORE_LIBRARY}
  ${QT_QTGUI_LIBRARY}
  ${QT_QTWEBKIT_LIBRARY}
  
  UNO23RegCore
  
  ORAIFMVC
  ORAIFImageAccess
  ORAIFImageTools
  ORAIFGUIComponents
  ORAIFModel
  
  QVTK
  vtkCommon
  vtkRendering
  vtkImaging
  vtkFiltering
  vtkGraphics
)

