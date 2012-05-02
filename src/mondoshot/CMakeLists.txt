##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_mondoshot)

## GCS: This won't build until dcmtk_uid.cxx is moved into base

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
include_directories (BEFORE ${src_plastimatch_SOURCE_DIR})
include_directories (BEFORE ${src_plastimatch_SOURCE_DIR}/base)
include_directories (BEFORE ${src_plastimatch_SOURCE_DIR}/sys)
include_directories (BEFORE ${CMAKE_BINARY_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${BSTRLIB_INCLUDE_DIR})
include_directories (AFTER ${LIBLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
include_directories (AFTER ${SQLITE_INCLUDE_DIR})
include_directories (AFTER ${SLICER_INCLUDE_DIRS}
  #${Slicer_Libs_INCLUDE_DIRS}
  #${Slicer_Base_INCLUDE_DIRS}
  )

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
set (MONDOSHOT_SRC
  dcmtk_uid.cxx dcmtk_uid.h
  mondoshot_dicom.cpp mondoshot_dicom.h
  mondoshot_main.cpp mondoshot_main.h
  )

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
# mondoshot requires WIN32, wx, dcmtk, sqlite3
if (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  add_executable (mondoshot WIN32 ${MONDOSHOT_SRC})
  target_link_libraries (mondoshot 
    ${wxWidgets_LIBRARIES} ${DCMTK_LIBRARIES} ${SQLITE_LIBRARIES})
  if (NOT BUILD_AGAINST_SLICER3)
    install (TARGETS mondoshot DESTINATION bin)
  endif ()
endif ()
