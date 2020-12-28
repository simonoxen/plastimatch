# - Findnlopt
# Find the native nlopt (NLopt) includes and library
#
#  nlopt_INCLUDE_DIR - where to find zlib.h, etc.
#  nlopt_LIBRARIES   - List of libraries when using zlib.
#  nlopt_FOUND       - True if zlib found.

set (nlopt_DIR "" CACHE PATH "Root of NLopt install tree (optional).")

if (nlopt_INCLUDE_DIR)
  # Already in cache, be silent
  set (nlopt_FIND_QUIETLY TRUE)
endif (nlopt_INCLUDE_DIR)

find_path (nlopt_INCLUDE_DIR nlopt.h
  ${nlopt_DIR}/include)

set (nlopt_NAMES nlopt nlopt_cxx)
find_library (nlopt_LIBRARY NAMES ${nlopt_NAMES}
  PATHS
  ${nlopt_DIR}/lib)

# handle the QUIETLY and REQUIRED arguments and set nlopt_FOUND to TRUE if 
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (nlopt DEFAULT_MSG 
  nlopt_LIBRARY 
  nlopt_INCLUDE_DIR)

if (nlopt_FOUND)
  set (nlopt_LIBRARIES ${nlopt_LIBRARY})
else ()
  set (nlopt_LIBRARIES)
endif ()

mark_as_advanced (nlopt_LIBRARY nlopt_INCLUDE_DIR)
