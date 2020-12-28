# CMake module to search for the libyaml library
# (library for parsing YAML files)
#
# If it's found it sets yaml_FOUND to TRUE
# and following variables are set:
# yaml_INCLUDE_DIR
# yaml_LIBRARY
find_path (yaml_INCLUDE_DIR NAMES yaml.h)
find_library (yaml_LIBRARIES NAMES yaml libyaml)
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (yaml
    DEFAULT_MSG yaml_LIBRARIES yaml_INCLUDE_DIR)
mark_as_advanced (yaml_INCLUDE_DIR yaml_LIBRARIES)
