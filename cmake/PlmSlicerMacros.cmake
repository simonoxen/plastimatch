# Load the Slicer configuration files
#
# Strangely, the Slicer use file (both Slicer 3 and Slicer 4) 
# appends the Slicer directories to our include path.  
# This causes nvcc.exe to barf on windows, so we can't let it do this.
#
# Equally strangely, in Slicer 4, the use file sets our C/CXX flags. 
# These cause nvcc + gcc-4.3 to barf because the flags are for gcc-4.5

macro (PLM_USE_SLICER)
  if (Slicer_USE_FILE)
    get_directory_property (OLD_INCLUDE_DIR INCLUDE_DIRECTORIES)
    set_directory_properties (PROPERTIES INCLUDE_DIRECTORIES "")
    set (OLD_CFLAGS ${CMAKE_C_FLAGS})
    set (OLD_CXXFLAGS ${CMAKE_CXX_FLAGS})
    
    include ("${Slicer_USE_FILE}")
    get_directory_property (SLICER_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    set_directory_properties (PROPERTIES INCLUDE_DIRECTORIES
      "${OLD_INCLUDE_DIR}")
    set (CMAKE_C_FLAGS "${OLD_CFLAGS}" CACHE STRING "CMake CXX Flags" FORCE)
    set (CMAKE_CXX_FLAGS "${OLD_CXXFLAGS}" CACHE STRING "CMake CXX Flags" FORCE)
  endif ()
  
  if (SLICER_IS_SLICER3)
    # Set reasonable default install prefix and output paths
    # (after setting Slicer3_DIR, delete CMAKE_INSTALL_PREFIX and re-configure)
    slicer3_set_default_install_prefix_for_external_projects ()
  else ()
    
    # Again, prevent Slicer 4 from overwriting C/CXX flags
    set (OLD_CFLAGS ${CMAKE_C_FLAGS})
    set (OLD_CXXFLAGS ${CMAKE_CXX_FLAGS})
    
    # 2012-01-10: JC says these aren't needed any more
    ## Slicer 4: include missing cmake scripts (Slicer3 stuff)
    include (Slicer3PluginsMacros)
    ## Slicer 4: include missing cmake scripts (loadable module stuff)
    include (vtkMacroKitPythonWrap)
    include (ctkMacroWrapPythonQt)
    include (ctkMacroCompilePythonScript)
    ## Slicer 4: this seems to be needed too
    include(${GenerateCLP_USE_FILE})
    
    set (CMAKE_C_FLAGS "${OLD_CFLAGS}" CACHE STRING "CMake CXX Flags" FORCE)
    set (CMAKE_CXX_FLAGS "${OLD_CXXFLAGS}" CACHE STRING "CMake CXX Flags" FORCE)
  endif ()
endmacro ()
