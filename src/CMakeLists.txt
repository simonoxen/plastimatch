##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src)

##-----------------------------------------------------------------------------
##  Figure out library dependencies
##-----------------------------------------------------------------------------
set (PLMSYS_LIBRARY_DEPENDENCIES 
  bstrlib
  )

set (GPUIT_LIBRARY_DEPENDENCIES 
  ${LIBLBFGS_LIBRARIES}
  nocedal
  plmsys
  )

if (NOT BUILD_DEBIAN)
  set (GPUIT_LIBRARY_DEPENDENCIES 
    ${GPUIT_LIBRARY_DEPENDENCIES}
    specfun
    )
endif ()

# JAS 2010.11.19
# New CUDA plugin stuff
if (CUDA_FOUND)
  if (PLM_USE_GPU_PLUGINS)      # << Use dynamic loading
    if (WIN32)
      # Windows
      set (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        plmcuda
        # shared library: cuda extensions
        # we tell windows to "delay load" this
      )
    else ()
      # Linux
      set (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        # In linux we use dlfcn.h to load plmcula
        # in the code @ runtime.
      )
    endif ()
  else ()    # << use static linking
    set (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      ${CUDA_LIBRARIES}
    )
  endif ()
endif ()

# JAS 2011.01.13
# New OpenCL plugin stuff (plmopencl)
if (OPENCL_FOUND)
  if (PLM_USE_GPU_PLUGINS)      # << Use dynamic loading
    if (WIN32)
      # Windows
      set (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        plmopencl
        # shared library: cuda extensions
        # we tell windows to "delay load" this
      )
    else ()
      # Linux
      set (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        # In linux we use delayload.h to load plmopencl
        # in the code @ runtime.
      )
    endif ()
  else ()    # << use static linking
    set (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      ${OPENCL_LIBRARIES}
    )
  endif ()
endif ()

if (FFTW_FOUND)
  set (GPUIT_LIBRARY_DEPENDENCIES 
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${FFTW_LIBRARIES})
endif ()

# Required - your choice: fortran or f2c
if (PLM_USE_F2C)
  set (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${F2C_LIBRARY}
    )
  if (UNIX)
    set (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      f2c_helper
      )
  endif ()
  # Apparently the below is not needed.
  # LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/libf2c)
else ()
  # It would seem that mixed-language programs need to set compiler and 
  # linker flags manually
  #   http://www.vtk.org/Wiki/CMakeForFortranExample
  # In theory, this is fixed in a future release
  #   http://www.cmake.org/Bug/view.php?id=9195
  message (STATUS "Fortran compiler = ${CMAKE_Fortran_COMPILER}")
  if (Fortran_COMPILER_NAME STREQUAL "gfortran" 
      OR Fortran_COMPILER_NAME STREQUAL "f95")
    set (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      gfortran
      )
  endif ()
endif ()

if (NOT HAVE_GETOPT_LONG)
  set (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    getopt
    )
endif ()

if (NLOPT_FOUND)
  set (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${NLOPT_LIBRARIES}
    )
endif ()

if (OPENMP_FOUND)
  set (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${OPENMP_LIBRARIES}
    )
endif ()

set (GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${MATH_LIB}
  )

if (LIBDL_FOUND)
  set (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    -ldl)
endif ()

set (GPUIT_LIBRARIES gpuit ${GPUIT_LIBRARY_DEPENDENCIES})
set (PLASTIMATCH_LIBS plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
set (SLICER_PLUGIN_LIBS ${ITK_LIBRARIES} ${GPUIT_LIBRARY_DEPENDENCIES})

if (PLM_CONFIG_ENABLE_LUA)
  set (PLASTIMATCH_LIBS
    ${PLASTIMATCH_LIBS}
    lua
    ${READLINE_LIBRARY}
    )
endif()

##-----------------------------------------------------------------------------
##  Determine linker flags
##    GPUIT.DLL should link with GPUIT_LIB_LDFLAGS
##    Executables that call PLMCUDA functions should link PLMCUDA_EXE_LDFLAGS
##    Other executables should link with GPUIT_EXE_LDFLAGS
##
##  JAS 2011.01.21
##    I have temporarily piggybacked OpenCL flags with CUDA flags.
##    As a result PLMCUDA_EXE_FLAGS should probably be renamed to
##    GPUPLUGIN_EXE_FLAGS or something of the like.  Probably unnecessary
##    to separate OpenCL and CUDA linker flags since inapplicable /DELAYLOAD
##    flags are automatically disregarded by the MS linker anyway.
##-----------------------------------------------------------------------------
set (GPUIT_LIB_LDFLAGS "${OPENMP_LDFLAGS}")
set (GPUIT_EXE_LDFLAGS "${OPENMP_LDFLAGS}")
set (PLMCUDA_EXE_LDFLAGS "${OPENMP_LDFLAGS}")
if (WIN32 AND NOT CYGWIN AND NOT MINGW)
  if (PLM_USE_GPU_PLUGINS)
    set (GPUIT_LIB_LDFLAGS 
      "${GPUIT_LIB_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:plmopencl.dll")
    set (PLMCUDA_EXE_LDFLAGS 
      "${PLMCUDA_EXE_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:plmopencl.dll")
  else ()
    set (GPUIT_LIB_LDFLAGS 
      "${GPUIT_LIB_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
    set (PLMCUDA_EXE_LDFLAGS 
      "${PLMCUDA_EXE_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
  endif ()
# else ()
  # if (LIBDL_FOUND)
  #   set (GPUIT_LIB_LDFLAGS "${GPUIT_LIB_LDFLAGS} -ldl")
  #   set (GPUIT_EXE_LDFLAGS "${GPUIT_EXE_LDFLAGS} -ldl")
  #   set (PLMCUDA_EXE_LDFLAGS "${PLMCUDA_EXE_LDFLAGS} -ldl")
  # endif ()
endif ()
message (STATUS "GPUIT_LIB_LDFLAGS: ${GPUIT_LIB_LDFLAGS}")
message (STATUS "GPUIT_EXE_LDFLAGS: ${GPUIT_EXE_LDFLAGS}")
message (STATUS "PLMCUDA_EXE_LDFLAGS: ${PLMCUDA_EXE_LDFLAGS}")

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
