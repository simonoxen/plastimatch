# - Find RTK
# Find the RTK includes and library
#
#  RTK_INCLUDE_DIR - where to find RTK header files
#  RTK_LIBRARIES   - List of libraries when using RTK
#  RTK_FOUND       - True if RTK found


IF (RTK_INCLUDE_DIR)
  # Already in cache, be silent
  SET (RTK_FIND_QUIETLY TRUE)
ENDIF (RTK_INCLUDE_DIR)

FIND_PATH (RTK_INCLUDE_DIR rtk.h)

SET (RTK_NAMES rtkIO)
FIND_LIBRARY (RTK_LIBRARY NAMES ${RTK_NAMES})

# handle the QUIETLY and REQUIRED arguments and set RTK_FOUND to TRUE if 
# all listed variables are TRUE

INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (RTK DEFAULT_MSG 
  RTK_LIBRARY 
  RTK_INCLUDE_DIR)

IF (RTK_FOUND)
  SET (RTK_LIBRARIES ${RTK_LIBRARY})
ELSE (RTK_FOUND)
  SET (RTK_LIBRARIES)
ENDIF (RTK_FOUND)

MARK_AS_ADVANCED (RTK_LIBRARY RTK_INCLUDE_DIR)
