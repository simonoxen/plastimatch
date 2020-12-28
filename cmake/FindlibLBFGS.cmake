# - Find libLBFGS
# Find the native libLBFGS includes and library
#
#  libLBFGS_INCLUDE_DIR - where to find the include files
#  libLBFGS_LIBRARY     - where to find the library file
#  libLBFGS_FOUND       - True if foudn


if (libLBFGS_INCLUDE_DIR)
  # Already in cache, be silent
  set (liblbfgs_FIND_QUIETLY TRUE)
endif ()

find_path (libLBFGS_INCLUDE_DIR lbfgs.h)

set (libLBFGS_NAMES lbfgs)
find_library (libLBFGS_LIBRARY NAMES ${libLBFGS_NAMES})

# handle the QUIETLY and REQUIRED arguments and set libLBFGS_FOUND to TRUE if 
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (libLBFGS DEFAULT_MSG 
  libLBFGS_LIBRARY 
  libLBFGS_INCLUDE_DIR)

mark_as_advanced (libLBFGS_LIBRARY libLBFGS_INCLUDE_DIR)
