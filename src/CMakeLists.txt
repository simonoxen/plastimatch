##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src)

##-----------------------------------------------------------------------------
##  Add subdirectories
##-----------------------------------------------------------------------------
if (PLM_CONFIG_DEBIAN_BUILD)
  set (PLM_CONFIG_DISABLE_ISE ON)
  set (PLM_CONFIG_DISABLE_FATM ON)
  set (PLM_CONFIG_DISABLE_MONDOSHOT ON)
  set (PLM_CONFIG_DISABLE_REG23 ON)
endif ()

if (NOT PLM_CONFIG_DISABLE_ISE)
  add_subdirectory (ise)
  set (PLM_CONFIG_DISABLE_FATM OFF)
endif ()

if (NOT PLM_CONFIG_DISABLE_FATM)
  add_subdirectory (fatm)
endif ()

# mondoshot requires plastimatch, WIN32, wx, dcmtk, sqlite3
if (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND
    AND NOT PLM_CONFIG_DISABLE_MONDOSHOT)
  add_subdirectory (mondoshot)
endif ()

if (NOT PLM_CONFIG_DISABLE_PLASTIMATCH)
  if (NOT ITK_FOUND)
    if (PLM_CONFIG_ENABLE_SUPERBUILD)
      set (proj plastimatch)
      if(CMAKE_EXTRA_GENERATOR)
	set(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
      else()
	set(gen "${CMAKE_GENERATOR}")
      endif()

      set (sb_option_list "")
      foreach (_var ${sb_cmake_vars})
	list (APPEND sb_option_list "-D${_var}=${${_var}}")
      endforeach ()
      ExternalProject_Add (${proj}
	DOWNLOAD_COMMAND ""
	INSTALL_COMMAND ""
	SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/plastimatch"
	BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/plastimatch"
	CMAKE_GENERATOR ${gen}
	CMAKE_ARGS
	-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
	-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
	-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
	-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
	-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
	-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
	-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
	-DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}
	-DITK_DIR:PATH=${ITK_DIR}
	-DDCMTK_DIR:PATH=${DCMTK_DIR}
	-DPLASTIMATCH_VERSION_STRING:STRING=${PLASTIMATCH_VERSION_STRING}
	${sb_option_list}
	DEPENDS
	ITK
	DCMTK
	devillard
	)
    else ()
      message (STATUS "Plastimatch will not be built (ITK not found)")
    endif ()
  else (NOT ITK_FOUND)
    add_subdirectory (plastimatch)
  endif (NOT ITK_FOUND)
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
if (PLM_BUILD_REG23 
    AND NOT ${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR} VERSION_LESS 4.0)
  message (STATUS "Reg-2-3 will not be built (wrong ITK version)")
  set (PLM_BUILD_ORAIFUTILS 0)
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23 AND ITK_USE_REVIEW_STATISTICS)
  message (STATUS 
    "Reg-2-3 will not be built (ITK compiled with review statistics)")
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
    AND ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION} VERSION_GREATER 5.10)
  message (STATUS "Reg-2-3 will not be built (wrong VTK version)")
  set (PLM_BUILD_ORAIFUTILS 0) ## REMOVE WHEN DONE
  set (PLM_BUILD_REG23 0)
endif ()
if (PLM_BUILD_REG23
    AND ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION} VERSION_LESS 5.6)
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
if (PLM_BUILD_REG23 
    AND ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR} VERSION_GREATER 4.8)
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

