##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
if (DCMTK_wrap_FIND_QUIETLY)
  list (APPEND DCMTK_EXTRA_ARGS QUIET)
endif ()
if (DCMTK_wrap_FIND_REQUIRED)
  list (APPEND DCMTK_EXTRA_ARGS REQUIRED)
endif ()

# if (DCMTK_DIR AND EXISTS "${DCMTK_DIR}/DCMTKConfig.cmake")
#   find_package (DCMTK NO_MODULE ${EXTRA_ARGS})
# else ()
#   find_package (DCMTK_legacy ${EXTRA_ARGS})
# endif ()

find_package (DCMTK NO_MODULE ${EXTRA_ARGS})
if (NOT DCMTK_FOUND)
  find_package (DCMTK_legacy ${EXTRA_ARGS})
endif ()
