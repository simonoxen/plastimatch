##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src)

##-----------------------------------------------------------------------------
##  PLASTIMATCH LIBRARIES
##-----------------------------------------------------------------------------
set (PLASTIMATCH_LIBS
  ${ITK_LIBRARIES}
  ${MATH_LIB}
  plmsys
  plmbase
  plmdose
  plmreconstruct
  plmregister
  plmsegment
  plmutil
  plmscript
  )

if (NOT BUILD_DEBIAN)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} specfun)
endif ()

if (FFTW_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${FFTW_LIBRARIES})
endif ()

if (NOT HAVE_GETOPT_LONG)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} getopt)
endif ()

if (NLOPT_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${NLOPT_LIBRARIES})
endif ()

if (OPENMP_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${OPENMP_LIBRARIES})
endif ()

if (LIBDL_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} -ldl)
endif ()

if (QT4_FOUND)
  set (PLASTIMATCH_LIBS ${PLASTIMATCH_LIBS} ${QT_LIBRARIES})
endif ()


##-----------------------------------------------------------------------------
##  SLICER LIBRARIES
##-----------------------------------------------------------------------------
set (SLICER_PLUGIN_LIBS ${PLASTIMATCH_LIBS})


##-----------------------------------------------------------------------------
##  DETERMINE LINKER FLAGS
##-----------------------------------------------------------------------------
set (PLASTIMATCH_LDFLAGS "${OPENMP_LDFLAGS}")
if (PLM_USE_GPU_PLUGINS AND WIN32 AND NOT CYGWIN AND NOT MINGW)
  set (PLASTIMATCH_LDFLAGS
  "${PLASTIMATCH_LDFLAGS} /DELAYLOAD:plmregistercuda.dll /DELAYLOAD:plmreconstructcuda.dll")
endif ()


##-----------------------------------------------------------------------------
##  Add subdirectories
##-----------------------------------------------------------------------------
if (PLM_BUILD_ISE)
  add_subdirectory (ise)
endif ()

if (NOT PLM_CONFIG_DISABLE_PLASTIMATCH)
  add_subdirectory (plastimatch)
  if (NOT BUILD_DEBIAN)
    add_subdirectory (slicer)
    if (NOT BUILD_AGAINST_SLICER3)
      add_subdirectory (fatm)
    endif ()
  endif ()
endif ()

set (PLM_BUILD_ORAIFUTILS 1)
set (PLM_BUILD_REG23 1)
if (PLM_CONFIG_DISABLE_REG23)
  set (PLM_BUILD_ORAIFUTILS 0)
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 
    AND ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 2.8)
  message (STATUS "Reg-2-3 will not be built (CMake version)")
  set (PLM_BUILD_ORAIFUTILS 0)
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND BUILD_AGAINST_SLICER3)
  message (STATUS "Reg-2-3 will not be built (build is with 3D-Slicer)")
  set (PLM_BUILD_ORAIFUTILS 0)
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND NOT ITK_FOUND)
  message (STATUS "Reg-2-3 will not be built (ITK not found)")
  set (PLM_BUILD_ORAIFUTILS 0)
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 
    AND ${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR} VERSION_LESS 3.20)
  message (STATUS "Reg-2-3 will not be built (wrong ITK version)")
  set (PLM_BUILD_ORAIFUTILS 0)
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND NOT VTK_FOUND)
  message (STATUS "Reg-2-3 will not be built (VTK not found)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND NOT VTK_USE_QT)
  message (STATUS "Reg-2-3 will not be built (VTK not built with QT)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 
    AND NOT ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION} VERSION_EQUAL 5.6)
  message (STATUS "Reg-2-3 will not be built (wrong VTK version)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND NOT QT_FOUND)
  message (STATUS "Reg-2-3 will not be built (QT not found)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 
    AND ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR} VERSION_LESS 4.6)
  message (STATUS "Reg-2-3 will not be built (wrong QT version)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND NOT QT_QMAKE_EXECUTABLE)
  message (STATUS "Reg-2-3 will not be built (no qmake executable)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND NOT QT_QTWEBKIT_FOUND)
  message (STATUS "Reg-2-3 will not be built (QtWebKit not found)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()

if (PLM_BUILD_ORAIFUTILS)
  add_subdirectory (oraifutils)
endif ()
if (PLM_BUILD_REG23)
  add_subdirectory (reg-2-3)
endif ()
