##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch "ISE" CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)
PROJECT(ise)

find_package (Bitflow)
find_package (MIL)

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_BINARY_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_BINARY_DIR})

##-----------------------------------------------------------------------------
## QT configuration
##-----------------------------------------------------------------------------
IF (QT4_FOUND AND QT_QTSQL_FOUND)
  SET (BUILD_ISE_QT TRUE)
ELSE ()
  SET (BUILD_ISE_QT FALSE)
ENDIF ()

##-----------------------------------------------------------------------------
##  VARIAN
##-----------------------------------------------------------------------------
FIND_PATH (VARIAN_L04_INCLUDE_DIR "HcpFuncDefs.h"
  "C:/Program Files/Varian/PaxscanL04/DeveloperFiles/Includes"
  )
FIND_LIBRARY (VARIAN_L04_LIBRARIES VirtCp 
  "C:/Program Files/Varian/PaxscanL04/DeveloperFiles/VirtCpRel")
IF (VARIAN_L04_INCLUDE_DIR AND VARIAN_L04_LIBRARIES)
  SET (VARIAN_L04_FOUND TRUE)
ELSE (VARIAN_L04_INCLUDE_DIR AND VARIAN_L04_LIBRARIES)
  SET (VARIAN_L04_FOUND FALSE)
ENDIF (VARIAN_L04_INCLUDE_DIR AND VARIAN_L04_LIBRARIES)

##-----------------------------------------------------------------------------
##  Advantech
##-----------------------------------------------------------------------------
FIND_PATH (ADVANTECH_INCLUDE_DIR "Driver.h"
  "C:/Program Files/Advantech/Adsapi/Include"
  )
FIND_PATH (ADVANTECH_LIBRARY_PATH "ADSDEV.lib"
  "C:/Program Files/Advantech/Adsapi/Lib"
  DOC "Path to ADVANTECH libraries.")
FIND_LIBRARY(ADVANTECH_ADS841_LIB ADS841 ${ADVANTECH_LIBRARY_PATH})
FIND_LIBRARY(ADVANTECH_Adsapi32_LIB Adsapi32 ${ADVANTECH_LIBRARY_PATH})
FIND_LIBRARY(ADVANTECH_Adsapi32bcb_LIB Adsapi32bcb ${ADVANTECH_LIBRARY_PATH})
FIND_LIBRARY(ADVANTECH_Adscomm_LIB Adscomm ${ADVANTECH_LIBRARY_PATH})
FIND_LIBRARY(ADVANTECH_ADSDEV_LIB ADSDEV ${ADVANTECH_LIBRARY_PATH})
FIND_LIBRARY(ADVANTECH_adsdnet_LIB adsdnet ${ADVANTECH_LIBRARY_PATH})

SET (ADVANTECH_LIBRARIES 
  ${ADVANTECH_Adscomm_LIB}
  ${ADVANTECH_Adsapi32_LIB}
  )
IF (ADVANTECH_INCLUDE_DIR AND ADVANTECH_LIBRARIES)
  SET (ADVANTECH_FOUND TRUE)
ELSE (ADVANTECH_INCLUDE_DIR AND ADVANTECH_LIBRARIES)
  SET (ADVANTECH_FOUND FALSE)
ENDIF (ADVANTECH_INCLUDE_DIR AND ADVANTECH_LIBRARIES)

##-----------------------------------------------------------------------------
##  Source files
##-----------------------------------------------------------------------------
SET (ISE_COMMON_SRC
  bitflow.c bitflow.h
  cbuf.c cbuf.h
  debug.c debug.h
  fileload.c fileload.h
  filewrite.c filewrite.h
  frame.c frame.h
  igpax.h
  indico_info.c indico_info.h
  ise_framework.c ise_framework.h
  ise.rc
  ise_error.h
  ise.c ise.h
  ise_gdi.c ise_gdi.h
  ise_gl.c ise_gl.h
  ise_gl_shader.c ise_gl_shader.h
  ise_globals.c ise_globals.h
  ise_igpax.c ise_igpax.h
  ise_main.c
  ise_ontrak.c ise_ontrak.h
  ise_rsc.h
  ise_sw_config.h
  ise_structs.h
  isewatch_client.c isewatch_client.h
  kill.c kill.h
  matrox_source.c matrox_source.h
  synthetic_source.c synthetic_source.h
  tracker.c tracker.h
  )

SET (ISE_SRC
  ${ISE_COMMON_SRC}
  ise_resize.c ise_resize.h
  )

SET (ISE_QT_SRC
  iqt_main.cxx
  iqt_main_window.cxx iqt_main_window.h
  iqt_video_widget.cxx iqt_video_widget.h
  )
IF (BUILD_ISE_QT)
  QT4_WRAP_CPP (ISE_QT_SRC 
    iqt_main_window.h 
    iqt_video_widget.h
    )
  QT4_WRAP_UI (ISE_QT_SRC 
    iqt_main_window.ui
    )
ENDIF ()

##-----------------------------------------------------------------------------
##  Configuration files
##-----------------------------------------------------------------------------
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/ise_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/ise_config.h)

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
IF(MIL_FOUND)
  INCLUDE_DIRECTORIES(AFTER ${MIL_INCLUDE_DIR})
ENDIF(MIL_FOUND)
IF(BITFLOW_FOUND)
  INCLUDE_DIRECTORIES(AFTER ${BITFLOW_INCLUDE_DIR})
ENDIF(BITFLOW_FOUND)
IF (VARIAN_L04_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${VARIAN_L04_INCLUDE_DIR})
ENDIF (VARIAN_L04_FOUND)
IF (ADVANTECH_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${ADVANTECH_INCLUDE_DIR})
ENDIF (ADVANTECH_FOUND)

## INCLUDE_DIRECTORIES(AFTER "C:/Program Files/NVIDIA Corporation/Cg/include")
## INCLUDE_DIRECTORIES(AFTER "C:/Program Files/fatm/include")
INCLUDE_DIRECTORIES (AFTER "${CMAKE_SOURCE_DIR}/src/fatm/include")

##-----------------------------------------------------------------------------
##  Linking
##-----------------------------------------------------------------------------
IF(MSVC80)
  LINK_DIRECTORIES("C:/Program Files/Microsoft Visual Studio 8/VC/lib")
ENDIF(MSVC80)
LINK_DIRECTORIES ("${CMAKE_BINARY_DIR}/src/fatm")
IF(MIL_FOUND)
  LINK_DIRECTORIES(${MIL_ROOT_DIR}/library/winnt/msc/dll)
ENDIF(MIL_FOUND)
IF(BITFLOW_FOUND)
  LINK_DIRECTORIES(${BITFLOW_ROOT_DIR}/lib)
ENDIF(BITFLOW_FOUND)

## LINK_DIRECTORIES("C:/Program Files/NVIDIA Corporation/Cg/lib")
## LINK_DIRECTORIES(${ise_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
IF (WIN32)
  ADD_EXECUTABLE(ise WIN32 ${ISE_SRC})
  TARGET_LINK_LIBRARIES(ise
    ${MIL_LIBRARIES}
    ${BITFLOW_LIBRARIES}
    AduHid
    cg cggl
    opengl32 glu32
    )
  INSTALL(TARGETS ise DESTINATION bin)
ENDIF ()

if (QT4_FOUND)
  set (ISE_QT_LIBRARIES ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
  plm_add_executable (ise_qt "${ISE_QT_SRC}" "${ISE_QT_LIBRARIES}" "" 0)
#  ADD_EXECUTABLE(ise_qt ${ISE_QT_SRC})
#  TARGET_LINK_LIBRARIES (ise_qt
#    ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
#  INSTALL(TARGETS ise_qt DESTINATION bin)
endif ()
