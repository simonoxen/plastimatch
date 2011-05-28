
# DRREngine library
SET(OWN_LIB_NAME ORAIFNReg2D3DDRREngine)

INCLUDE_DIRECTORIES(
  ${ORAIFUTILS_SOURCE_DIR}/${ORAIFUTILS_NREG2D3D_MATHS_SUBDIR} 
  ${ORAIFUTILS_SOURCE_DIR}/${ORAIFUTILS_NREG2D3D_TRANSFORM_SUBDIR}
)

SET(${OWN_LIB_NAME}_HDRS
  oraGLSLDRRRayCasterDevMethods.hxx
  oraGLSLDRRRayCastMapper.h
  oraGLSLRayCastingCodeFragments.hxx
  
  oraRGBTableManager.h
  
  oraScalarFieldManager.h
  
  oraITKVTKDRRFilter.h
  oraITKVTKDRRFilter.txx
  
  oraProjectionProperties.h
  oraProjectionProperties.txx
  oraFlexMapCorrection.h
  oraFlexMapCorrection.txx
  oraTranslationScaleFlexMapCorrection.h
  oraTranslationScaleFlexMapCorrection.txx
  oraLinacMountedImagingDeviceProjectionProperties.h
  oraLinacMountedImagingDeviceProjectionProperties.txx
  oraPortalImagingDeviceProjectionProperties.h
  oraPortalImagingDeviceProjectionProperties.txx
)

SET(${OWN_LIB_NAME}_SRCS
  oraGLSLDRRRayCastMapper.cxx  
  oraRGBTableManager.cxx
  oraScalarFieldManager.cxx
)

# Convince MSVC that txx files are also C++ files!
IF(MSVC)
  SET_SOURCE_FILES_PROPERTIES(
    oraITKVTKDRRFilter.txx
    oraProjectionProperties.txx
    oraFlexMapCorrection.txx
    oraTranslationScaleFlexMapCorrection.txx
    oraLinacMountedImagingDeviceProjectionProperties.txx
    oraPortalImagingDeviceProjectionProperties.txx
    PROPERTIES 
    LANGUAGE "CXX" 
    HEADER_FILE_ONLY FALSE
    COMPILE_FLAGS ""
  )
ENDIF(MSVC)

# Support for invisible X windows on X environments. 
IF (UNIX AND NOT WIN32)
  SET(${OWN_LIB_NAME}_HDRS ${${OWN_LIB_NAME}_HDRS}
    oraInvisibleXOpenGLRenderWindow.h
  )
  
  SET(${OWN_LIB_NAME}_SRCS ${${OWN_LIB_NAME}_SRCS}
    oraInvisibleXOpenGLRenderWindow.cxx
  )
ENDIF(UNIX AND NOT WIN32)

ADD_LIBRARY(${OWN_LIB_NAME} ${${OWN_LIB_NAME}_SRCS} ${${OWN_LIB_NAME}_HDRS})
SET_TARGET_PROPERTIES(${OWN_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
TARGET_LINK_LIBRARIES(${OWN_LIB_NAME} 
  ITKCommon ITKIO ITKAlgorithms ITKBasicFilters ITKNumerics
  vtkCommon vtkVolumeRendering vtkFiltering vtkRendering
)

# Add headers to export-set. NOTE: Quotes are important here otherwise only the first header is installed!
SET_TARGET_PROPERTIES(${OWN_LIB_NAME} PROPERTIES PUBLIC_HEADER "${${OWN_LIB_NAME}_HDRS}")

INSTALL(TARGETS ${OWN_LIB_NAME}
  # Add the library to the export-set
  EXPORT ORAIFUTILSLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib/static
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/${ORAIFUTILS_NREG2D3D_DRRENGINE_SUBDIR}"
  COMPONENT dev)

# UNIT TEST
IF (ORAIFUTILS_BUILD_TESTING)
  ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/testing)
ENDIF (ORAIFUTILS_BUILD_TESTING)
