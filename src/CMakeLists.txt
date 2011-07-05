##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
PROJECT (src)

##-----------------------------------------------------------------------------
##  Figure out library dependencies
##-----------------------------------------------------------------------------
SET (PLMSYS_LIBRARY_DEPENDENCIES 
  bstrlib
  )

SET (GPUIT_LIBRARY_DEPENDENCIES 
  ${LIBLBFGS_LIBRARIES}
  nocedal
  plmsys
  specfun
  )

# Optional
IF (BROOK_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${BROOK_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${DIRECTX_LIBRARIES}
    )
ENDIF ()

# JAS 2010.11.19
# New CUDA plugin stuff
IF (CUDA_FOUND)
  IF (PLM_USE_GPU_PLUGINS)      # << Use dynamic loading
    IF (WIN32)
      # Windows
      SET (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        plmcuda
        # shared library: cuda extensions
        # we tell windows to "delay load" this
      )
    ELSE ()
      # Linux
      SET (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        # In linux we use dlfcn.h to load plmcula
        # in the code @ runtime.
      )
    ENDIF ()
  ELSE ()    # << use static linking
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      ${CUDA_LIBRARIES}
    )
  ENDIF ()
ENDIF ()

# JAS 2011.01.13
# New OpenCL plugin stuff (plmopencl)
IF (OPENCL_FOUND)
  IF (PLM_USE_GPU_PLUGINS)      # << Use dynamic loading
    IF (WIN32)
      # Windows
      SET (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        plmopencl
        # shared library: cuda extensions
        # we tell windows to "delay load" this
      )
    ELSE ()
      # Linux
      SET (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        # In linux we use delayload.h to load plmopencl
        # in the code @ runtime.
      )
    ENDIF ()
  ELSE ()    # << use static linking
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      ${OPENCL_LIBRARIES}
    )
  ENDIF ()
ENDIF ()

IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES 
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${FFTW_LIBRARIES})
ENDIF ()

# Required - your choice: fortran or f2c
IF (PLM_USE_F2C)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${F2C_LIBRARY}
    )
  IF (UNIX)
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      f2c_helper
      )
  ENDIF ()
  # Apparently the below is not needed.
  # LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/libf2c)
ELSE ()
  # It would seem that mixed-language programs need to set compiler and 
  # linker flags manually
  #   http://www.vtk.org/Wiki/CMakeForFortranExample
  # In theory, this is fixed in a future release
  #   http://www.cmake.org/Bug/view.php?id=9195
  MESSAGE (STATUS "Fortran compiler = ${CMAKE_Fortran_COMPILER}")
  IF (Fortran_COMPILER_NAME STREQUAL "gfortran" 
      OR Fortran_COMPILER_NAME STREQUAL "f95")
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      gfortran
      )
  ENDIF ()
ENDIF ()

IF (NOT HAVE_GETOPT_LONG)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    getopt
    )
ENDIF ()

IF (NLOPT_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${NLOPT_LIBRARIES}
    )
ENDIF ()

IF (OPENMP_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${OPENMP_LIBRARIES}
    )
ENDIF ()

SET (GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${MATH_LIB}
  )

SET (GPUIT_LIBRARIES gpuit ${GPUIT_LIBRARY_DEPENDENCIES})
SET (PLASTIMATCH_LIBS plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
SET (SLICER_PLUGIN_LIBS ${ITK_LIBRARIES} ${GPUIT_LIBRARY_DEPENDENCIES})

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
SET (GPUIT_LIB_LDFLAGS "${OPENMP_LDFLAGS}")
SET (GPUIT_EXE_LDFLAGS "${OPENMP_LDFLAGS}")
SET (PLMCUDA_EXE_LDFLAGS "${OPENMP_LDFLAGS}")
IF (WIN32 AND NOT CYGWIN AND NOT MINGW)
  IF (PLM_USE_GPU_PLUGINS)
    SET (GPUIT_LIB_LDFLAGS 
      "${GPUIT_LIB_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:plmopencl.dll")
    SET (PLMCUDA_EXE_LDFLAGS 
      "${PLMCUDA_EXE_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:plmopencl.dll")
  ELSE ()
    SET (GPUIT_LIB_LDFLAGS 
      "${GPUIT_LIB_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
    SET (PLMCUDA_EXE_LDFLAGS 
      "${PLMCUDA_EXE_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
  ENDIF ()
ELSE ()
  IF (LIBDL_FOUND)
    SET (GPUIT_LIB_LDFLAGS "${GPUIT_LIB_LDFLAGS} -ldl")
    SET (GPUIT_EXE_LDFLAGS "${GPUIT_EXE_LDFLAGS} -ldl")
    SET (PLMCUDA_EXE_LDFLAGS "${PLMCUDA_EXE_LDFLAGS} -ldl")
  ENDIF ()
ENDIF ()
MESSAGE (STATUS "GPUIT_LIB_LDFLAGS: ${GPUIT_LIB_LDFLAGS}")
MESSAGE (STATUS "GPUIT_EXE_LDFLAGS: ${GPUIT_EXE_LDFLAGS}")
MESSAGE (STATUS "PLMCUDA_EXE_LDFLAGS: ${PLMCUDA_EXE_LDFLAGS}")

##-----------------------------------------------------------------------------
##  Add subdirectories
##-----------------------------------------------------------------------------
add_subdirectory (plastimatch)

if (NOT BUILD_DEBIAN)
  add_subdirectory (slicer)
  if (NOT BUILD_AGAINST_SLICER3)
    add_subdirectory (fatm)
    if (PLM_BUILD_ISE)
      add_subdirectory (ise)
    endif ()
  endif ()
endif ()


if (PLM_BUILD_REG23
    AND NOT BUILD_AGAINST_SLICER3
    AND NOT ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} VERSION_LESS 2.8
    AND VTK_FOUND 
    AND VTK_USE_QT
    AND ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION} VERSION_EQUAL 5.6
    AND ITK_FOUND
    AND NOT ${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR} VERSION_LESS 3.20
    AND (MINGW OR ( QT_FOUND
    AND NOT ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR} VERSION_LESS 4.6 )
    OR QT_QMAKE_EXECUTABLE)
    )
  add_subdirectory (oraifutils)
  add_subdirectory (reg-2-3)
endif ()
