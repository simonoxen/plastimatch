##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
if (DCMTK_wrap_FIND_QUIETLY)
  list (APPEND DCMTK_EXTRA_ARGS QUIET)
endif ()
if (DCMTK_wrap_FIND_REQUIRED)
  list (APPEND DCMTK_EXTRA_ARGS REQUIRED)
endif ()

find_package (DCMTK NO_MODULE ${DCMTK_EXTRA_ARGS})
if (NOT DCMTK_FOUND)
  message (STATUS "Searching for DCMTK using legacy method")
  find_package (DCMTK_legacy ${DCMTK_EXTRA_ARGS})
endif ()

# The DCMTK 3.6.2 DCMTKConfig.cmake seems to be broken on windows
string (REPLACE "DCMTK_INCLUDE_DIRS-NOTFOUND;" "" DCMTK_INCLUDE_DIRS
  "${DCMTK_INCLUDE_DIRS}")

# Additional test for compatibility across DCMTK versions
if (DCMTK_FOUND)
  include (CheckDCMTK)
endif ()
