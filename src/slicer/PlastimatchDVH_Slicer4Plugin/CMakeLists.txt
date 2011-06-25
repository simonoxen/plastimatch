#-----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8.2)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# See http://cmake.org/cmake/help/cmake-2-8-docs.html#section_Policies for details
#-----------------------------------------------------------------------------
if (POLICY CMP0016)
  cmake_policy (SET CMP0016 NEW)
endif ()
if (POLICY CMP0017)
  cmake_policy (SET CMP0017 OLD)
endif ()

# Module name (usually identique to the directory name)
set (EXTENSION_NAME PlastimatchDVH)

if (NOT Slicer_SOURCE_DIR)

  # GCS: 2011-06-24.  What are all these for???
  
  #SET(EXTENSION_HOMEPAGE "http://www.slicer.org/slicerWiki/index.php/Slicer4:Developers:Projects:QtSlicer/Tutorials/ExtensionWriting")
  #SET(EXTENSION_CATEGORY "Examples")
  #SET(EXTENSION_STATUS "Beta")
  #SET(EXTENSION_DESCRIPTION "This is an example of Qt loadable module built as an extension")
  #SET(ADDITIONAL_C_FLAGS "" CACHE STRING "Additional ${EXTENSION_NAME} C Flags")
  #SET(ADDITIONAL_CXX_FLAGS "" CACHE STRING "Additional ${EXTENSION_NAME} CXX Flags")

  # GCS: 2011-06-24.  This cannot be omitted, even though the use file 
  # was already included by the parent.  
  INCLUDE(${Slicer_USE_FILE})
  
  #SET(EXTENSION_LICENSE_FILE ${Slicer_LICENSE_FILE})
  #SET(EXTENSION_README_FILE ${Slicer_README_FILE})

  # GCS: 2011-06-24.  This makes an "Experimental" target, even though
  # the parent has already created an "Experimental" target.
  # INCLUDE(SlicerEnableExtensionTesting)

endif ()

if (BUILD_TESTING)
  add_subdirectory (Testing)
endif ()

# Add subdirectories
add_subdirectory (Logic)

set (qt_module_export_directive "Q_SLICER_QTMODULES_PLASTIMATCHDVH_EXPORT")

# Additional includes - Current_{source,binary} and Slicer_{Libs,Base} already included
set (qt_module_include_directories
  ${CMAKE_CURRENT_SOURCE_DIR}/Logic
  ${CMAKE_CURRENT_BINARY_DIR}/Logic
  )

# Source files
set (qt_module_SRCS
  qSlicerPlastimatchDVHModule.cxx
  qSlicerPlastimatchDVHModule.h
  qSlicerPlastimatchDVHModuleWidget.cxx
  qSlicerPlastimatchDVHModuleWidget.h
  )

# Headers that should run through moc
set (qt_module_MOC_SRCS
  qSlicerPlastimatchDVHModule.h
  qSlicerPlastimatchDVHModuleWidget.h
  )

# UI files
set (qt_module_UI_SRCS
  Resources/UI/qSlicerPlastimatchDVHModule.ui
)

# Additional Target libraries
set (qt_module_target_libraries
  vtkSlicerPlastimatchDVHModuleLogic
  )

# Resources
set (qt_module_resources
  Resources/qSlicerPlastimatchDVHModule.qrc
)

set (qt_module_target_libraries
  ${qt_module_target_libraries}
  plastimatch1
  gpuit
  plmsys
  )

slicerMacroBuildQtModule(
  NAME ${EXTENSION_NAME}
  EXPORT_DIRECTIVE ${qt_module_export_directive}
  INCLUDE_DIRECTORIES ${qt_module_include_directories}
  SRCS ${qt_module_SRCS}
  MOC_SRCS ${qt_module_MOC_SRCS}
  UI_SRCS ${qt_module_UI_SRCS}
  TARGET_LIBRARIES ${qt_module_target_libraries} plastimatch1 gpuit plmsys
  RESOURCES ${qt_module_resources}
  )

# GCS: 2011-06-24.  Two bugs.
# (1) SlicerFunctionGenerateExtensionDescription.cmake pulls in 
#     requirement that git must be installed.
# (2) FindGit.cmake does not search for cygwin git.
# IF(NOT Slicer_SOURCE_DIR)
#  INCLUDE(SlicerFunctionGenerateExtensionDescription)
#    slicerFunctionGenerateExtensionDescription(
#    EXTENSION_NAME ${EXTENSION_NAME}
#    EXTENSION_CATEGORY ${EXTENSION_CATEGORY} 
#    EXTENSION_STATUS ${EXTENSION_STATUS}
#    EXTENSION_HOMEPAGE ${EXTENSION_HOMEPAGE}
#    EXTENSION_DESCRIPTION ${EXTENSION_DESCRIPTION}
#    DESTINATION_DIR ${CMAKE_CURRENT_BINARY_DIR}
#    SLICER_WC_REVISION ${Slicer_WC_REVISION}
#    SLICER_WC_ROOT ${Slicer_WC_ROOT}
#  )
#  INCLUDE(${Slicer_EXTENSION_CPACK})
# ENDIF()
