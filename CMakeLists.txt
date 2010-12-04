##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
PROJECT (plastimatch)

CMAKE_MINIMUM_REQUIRED (VERSION 2.6.0)

## Tell CMake to be quiet
CMAKE_POLICY (SET CMP0003 NEW)

## Default to release
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_BUILD_TYPE)

## My modules override CMake defaults
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

##-----------------------------------------------------------------------------
##  Plastimatch configuration variables
##-----------------------------------------------------------------------------
# Building little test programs
SET (PLM_BUILD_CUDA_TEST OFF CACHE BOOL "Build CUDA test program")
SET (PLM_BUILD_FANN_TEST OFF CACHE BOOL "Build fann test program")
SET (PLM_BUILD_MEX_TEST OFF CACHE BOOL "Build mex test program")
SET (PLM_BUILD_NLOPT_TEST OFF CACHE BOOL "Build NLopt test program")
SET (PLM_BUILD_OPENCL_TEST OFF CACHE BOOL "Build OpenCL test program")
SET (PLM_BUILD_OPENMP_TEST OFF CACHE BOOL "Build OpenMP test program")
SET (PLM_BUILD_QT_TEST OFF CACHE BOOL "Build Qt test program")
SET (PLM_BUILD_RTK OFF CACHE BOOL "Build RTK")

# Software options
SET (PLM_USE_4D_SS_IMAGE OFF CACHE BOOL 
  "Save structure sets as 4D ITK UCHAR images (in development)")
SET (PLM_SLICER_HARDCODED_FILENAME OFF CACHE BOOL 
  "Use hardcoded filename instead of tmpfile()")

# Compile and link options
SET (PLM_USE_CUDA_PLUGIN OFF CACHE BOOL
  "Build CUDA support as a runtime loaded shared library")
SET (PLM_BUILD_SLICER_LOADABLE OFF CACHE BOOL 
  "Build slicer loadable module")
SET (PLM_INSTALL_RPATH ON CACHE BOOL 
  "Add full RPATH to install")
SET (PLM_BUILD_SHARED_LIBS OFF CACHE BOOL 
  "Build plastimatch as shared library")
SET (PLM_MONOLITHIC_LIBRARY OFF CACHE BOOL 
  "Link gpuit and plastimatch1 together")
SET (PLM_PREFER_F2C OFF CACHE BOOL 
  "Prefer using f2c over native fortran compiler")
SET (PLM_PREFER_SYSTEM_F2C ON CACHE BOOL 
  "Prefer the system f2c over the included f2c")

# Regression testing
SET (PLM_BUILD_TESTING ON CACHE BOOL 
  "Enable regression testing")
SET (PLM_REDUCED_TESTS OFF CACHE BOOL 
  "Run fewer tests (used when debugging tests)")
SET (PLM_TEST_BSPLINE_FLAVORS OFF CACHE BOOL 
  "Regression tests for seldom used b-spline implementations")
SET (PLM_TEST_DICOMRT_XIO OFF CACHE BOOL 
  "Regression test for dicom, dicomrt (downloads extra test data)")
SET (PLM_TEST_DICOMRT_PINNACLE OFF CACHE BOOL 
  "Regression test for dicom, dicomrt (downloads extra test data)")
SET (PLM_TEST_FDK_VARIAN OFF CACHE BOOL 
  "Regression tests for fdk reconstruction from Varian data (downloads extra test data)")
SET (PLM_TEST_XIO OFF CACHE BOOL 
  "Regression tests for xio file exchange (downloads extra test data)")

##-----------------------------------------------------------------------------
##  Extra paths
##-----------------------------------------------------------------------------
SET (PLM_TESTING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Testing")
SET (PLM_TESTING_DATA_DIR "${CMAKE_SOURCE_DIR}/Testing/Data")
SET (PLM_TESTING_BUILD_DIR "${CMAKE_BINARY_DIR}/Testing")

##-----------------------------------------------------------------------------
##  Include files
##-----------------------------------------------------------------------------
INCLUDE (CTest)
INCLUDE (CheckFunctionExists)
INCLUDE (CheckIncludeFiles)
INCLUDE (CheckTypeSize)
INCLUDE (TestBigEndian)

# http://www.cmake.org/pipermail/cmake/2008-December/025886.html
INCLUDE (InstallRequiredSystemLibraries)

# http://www.cmake.org/Bug/view.php?id=9220
INCLUDE (language_support.cmake)

##-----------------------------------------------------------------------------
##  Macros for creating targets
##-----------------------------------------------------------------------------
MACRO (PLM_ADD_LIBRARY 
    TARGET_NAME TARGET_SRC TARGET_LIBS)

  IF (BUILD_AGAINST_SLICER3)
    ADD_LIBRARY (${TARGET_NAME} STATIC ${TARGET_SRC})
    SLICER3_SET_PLUGINS_OUTPUT_PATH (${TARGET_NAME})
  ELSE (BUILD_AGAINST_SLICER3)
    ADD_LIBRARY (${TARGET_NAME} ${TARGET_SRC})
    INSTALL (TARGETS ${TARGET_NAME}
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib)
  ENDIF (BUILD_AGAINST_SLICER3)
  TARGET_LINK_LIBRARIES (${TARGET_NAME} ${TARGET_LIBS})
ENDMACRO (PLM_ADD_LIBRARY)
    
MACRO (PLM_ADD_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  IF (NOT BUILD_AGAINST_SLICER3)
    ADD_EXECUTABLE(${TARGET_NAME} ${TARGET_SRC})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} ${TARGET_LIBS})
    IF (NOT ${TARGET_LDFLAGS} STREQUAL "")
      SET_TARGET_PROPERTIES(${TARGET_NAME} 
	PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
    ENDIF (NOT ${TARGET_LDFLAGS} STREQUAL "")
    # CXX linkage required for nlopt
    SET_TARGET_PROPERTIES (${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    INSTALL(TARGETS ${TARGET_NAME} DESTINATION bin)
  ENDIF (NOT BUILD_AGAINST_SLICER3)
ENDMACRO (PLM_ADD_EXECUTABLE)

MACRO (PLM_ADD_SLICER_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  GENERATECLP (${TARGET_SRC} ${TARGET_NAME}.xml)
  ADD_EXECUTABLE (${TARGET_NAME} ${TARGET_SRC})
  TARGET_LINK_LIBRARIES (${TARGET_NAME} ${TARGET_LIBS})
  IF (NOT ${TARGET_LDFLAGS} STREQUAL "")
    SET_TARGET_PROPERTIES (${TARGET_NAME} 
      PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
  ENDIF (NOT ${TARGET_LDFLAGS} STREQUAL "")
  SLICER3_SET_PLUGINS_OUTPUT_PATH (${TARGET_NAME})
  SLICER3_INSTALL_PLUGINS (${TARGET_NAME})
ENDMACRO (PLM_ADD_SLICER_EXECUTABLE)

MACRO (PLM_ADD_SLICER_MODULE 
    TARGET_NAME TARGET_SRC TARGET_LIBS)

  #GENERATELM (TARGET_SRC ${TARGET_NAME}.xml)
  ADD_LIBRARY (${TARGET_NAME} ${TARGET_SRC})
  TARGET_LINK_LIBRARIES (${TARGET_NAME} 
    ${Slicer_Libs_LIBRARIES}
    ${Slicer_Base_LIBRARIES}
    ${KWWidgets_LIBRARIES}
    ${ITK_LIBRARIES}
    ${TARGET_LIBS})

  SLICER3_SET_MODULES_OUTPUT_PATH (${TARGET_NAME})
  #SLICER3_INSTALL_PLUGINS (${TARGET_NAME})
ENDMACRO (PLM_ADD_SLICER_MODULE)

MACRO (PLM_ADD_OPENCL_FILE SRCS CL_FILE)
  ## I don't yet know how to bundle the .cl file within the executable.
  ## Therefore, copy the .cl into binary directory.
  SET (${SRCS} ${${SRCS}} "${CMAKE_BINARY_DIR}/${CL_FILE}")
  ADD_CUSTOM_COMMAND (
    OUTPUT "${CMAKE_BINARY_DIR}/${CL_FILE}"
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_SOURCE_DIR}/${CL_FILE}" 
    "${CMAKE_BINARY_DIR}/${CL_FILE}" 
    DEPENDS "${CMAKE_SOURCE_DIR}/${CL_FILE}")
  ## Need in the testing directory too :(
  SET (${SRCS} ${${SRCS}} "${PLM_TESTING_BUILD_DIR}/${CL_FILE}")
  ADD_CUSTOM_COMMAND (
    OUTPUT "${PLM_TESTING_BUILD_DIR}/${CL_FILE}"
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_SOURCE_DIR}/${CL_FILE}" 
    "${PLM_TESTING_BUILD_DIR}/${CL_FILE}" 
    DEPENDS "${CMAKE_SOURCE_DIR}/${CL_FILE}")
ENDMACRO (PLM_ADD_OPENCL_FILE)

##-----------------------------------------------------------------------------
##  Search for include files
##-----------------------------------------------------------------------------
CHECK_INCLUDE_FILES ("sys/stat.h" HAVE_SYS_STAT_H)

##-----------------------------------------------------------------------------
##  Set the math library -- need this variable to configure libf2c
##-----------------------------------------------------------------------------
IF (UNIX)
  SET (MATH_LIB -lm)
ELSE (UNIX)
  SET (MATH_LIB)
ENDIF (UNIX)

##-----------------------------------------------------------------------------
##  Search for languages and language features
##-----------------------------------------------------------------------------

# C/C++ 32/64
CHECK_TYPE_SIZE ("unsigned int"    CMAKE_SIZEOF_UINT)
CHECK_TYPE_SIZE ("unsigned long"   CMAKE_SIZEOF_ULONG)
TEST_BIG_ENDIAN (CMAKE_WORDS_BIGENDIAN)

# Always check for fortran, whether we use it or not
IF (NOT MINGW)
  # For some reason, CMake tries to invoke devenv instead of gfortran
  # for mingw.
  INCLUDE (PLM_FindFortran.cmake)

  # The FortranCInterface is not included on CMake 2.6.2
  # INCLUDE (FortranCInterface)
ENDIF (NOT MINGW)

# Choose between f2c or fortran
SET (PLM_USE_F2C FALSE)
IF (PLM_PREFER_F2C OR NOT CMAKE_Fortran_COMPILER_WORKS)
  SET (PLM_USE_F2C TRUE)
ENDIF (PLM_PREFER_F2C OR NOT CMAKE_Fortran_COMPILER_WORKS)

IF (PLM_USE_F2C)
  # If the user wants the system f2c, check for it
  IF (PLM_PREFER_SYSTEM_F2C)
    FIND_PACKAGE (F2C)
  ENDIF (PLM_PREFER_SYSTEM_F2C)

  # If the user doesn't want system f2c, or didn't find it, build the 
  # included f2c
  IF (NOT PLM_PREFER_SYSTEM_F2C OR NOT F2C_LIBRARY)
    SET (PLM_USE_INCLUDED_F2C 1)
    ADD_SUBDIRECTORY (libs/libf2c)
    SET (F2C_LIBRARY f2c1)
    SET (F2C_INCLUDE_DIR libs/libf2c)
  ENDIF (NOT PLM_PREFER_SYSTEM_F2C OR NOT F2C_LIBRARY)
ENDIF (PLM_USE_F2C)

##-----------------------------------------------------------------------------
##  Search for libraries
##-----------------------------------------------------------------------------
FIND_PACKAGE (Brook)
FIND_PACKAGE (DCMTK)
FIND_PACKAGE (Etags)
FIND_PACKAGE (Fann)
FIND_PACKAGE (FFTW)
FIND_PACKAGE (Gengetopt)
FIND_PACKAGE (Liblbfgs)
FIND_PACKAGE (NLopt)
IF (PLM_BUILD_MEX_TEST)
  FIND_PACKAGE (Matlab)
ENDIF (PLM_BUILD_MEX_TEST)
FIND_PACKAGE (Octave)
FIND_PACKAGE (OpenCL)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OpenMP)
FIND_PACKAGE (Qt4 4.6.2 COMPONENTS QtCore QtGui OPTIONAL)
IF (PLM_BUILD_RTK)
  FIND_PACKAGE (RTK)
ENDIF (PLM_BUILD_RTK)
FIND_PACKAGE (SQLite)
FIND_PACKAGE (SSE)           # SSE Extensions for CPU
FIND_PACKAGE (Subversion)
FIND_PACKAGE (wxWidgets)

##-----------------------------------------------------------------------------
##  QT requires including the use file
##-----------------------------------------------------------------------------
IF (QT4_FOUND)
  MESSAGE (STATUS "Looking for Qt4 - found")
  INCLUDE (${QT_USE_FILE})
ELSE (QT4_FOUND)
  MESSAGE (STATUS "Looking for Qt4 - not found")
ENDIF (QT4_FOUND)

##-----------------------------------------------------------------------------
##  Use local sqlite if not found
##-----------------------------------------------------------------------------
IF (NOT SQLITE_FOUND)
  ADD_SUBDIRECTORY (libs/sqlite-3.6.21)
  SET (SQLITE_INCLUDE_DIR libs/sqlite-3.6.21)
  SET (SQLITE_LIBRARIES sqlite3)
ENDIF (NOT SQLITE_FOUND)

##-----------------------------------------------------------------------------
##  Use local liblbfgs if not found
##-----------------------------------------------------------------------------
IF (NOT LIBLBFGS_FOUND)
  ADD_SUBDIRECTORY (libs/liblbfgs-1.9)
  SET (LIBLBFGS_INCLUDE_DIR libs/liblbfgs-1.9/include)
  SET (LIBLBFGS_LIBRARIES lbfgs)
  LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/liblbfgs-1.9)
  SET (LIBLBFGS_FOUND TRUE)
ENDIF (NOT LIBLBFGS_FOUND)

##-----------------------------------------------------------------------------
##  Only use local bstrlib
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (libs/bstrlib-05122010)
SET (BSTRLIB_INCLUDE_DIR libs/bstrlib-05122010)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/bstrlib-05122010)

##-----------------------------------------------------------------------------
##  Only use local nocedal
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (libs/nocedal)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/nocedal)

##-----------------------------------------------------------------------------
##  Only use local specfun
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (libs/specfun)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/specfun)

##-----------------------------------------------------------------------------
##  Special CUDA processing
##-----------------------------------------------------------------------------
FIND_PACKAGE (CUDA_wrap)
SET (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

IF (CUDA_FOUND)
  CUDA_INCLUDE_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
ENDIF (CUDA_FOUND)

##-----------------------------------------------------------------------------
##  Slicer
##-----------------------------------------------------------------------------
FIND_PACKAGE (Slicer QUIET)
IF (SLICER_FOUND)
  IF (Slicer_USE_FILE)
    # Include USE file.  Strangely, the use file appends the 
    # Slicer directories to our include path.  This causes nvcc.exe 
    # to barf on windows, so we can't do this.
    GET_DIRECTORY_PROPERTY (OLD_INCLUDE_DIR INCLUDE_DIRECTORIES)
    SET_DIRECTORY_PROPERTIES (PROPERTIES INCLUDE_DIRECTORIES "")
    INCLUDE ("${Slicer_USE_FILE}")
    GET_DIRECTORY_PROPERTY (SLICER_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    SET_DIRECTORY_PROPERTIES (PROPERTIES INCLUDE_DIRECTORIES
      "${OLD_INCLUDE_DIR}")
  ENDIF (Slicer_USE_FILE)

  # Set reasonable default install prefix and output paths
  # (after setting Slicer3_DIR, delete CMAKE_INSTALL_PREFIX and re-configure)
  slicer3_set_default_install_prefix_for_external_projects ()
  
  #slicer3_enable_plugins_testing()

  # Slicer seems to require a shared library build
  SET (PLM_BUILD_SHARED_LIBS ON)
ELSE (SLICER_FOUND)
  MESSAGE (STATUS "Looking for Slicer3 - not found")
ENDIF (SLICER_FOUND)


##-----------------------------------------------------------------------------
##  Special linking instructions on unix
##  http://www.cmake.org/Wiki/CMake_RPATH_handling
##-----------------------------------------------------------------------------
IF (PLM_INSTALL_RPATH)
  # use, i.e. don't skip the full RPATH for the build tree
  SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

  # the RPATH to be used when installing
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
ENDIF (PLM_INSTALL_RPATH)

##-----------------------------------------------------------------------------
##  User feedback on which languages and libraries were found
##-----------------------------------------------------------------------------
IF (wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - found.")
ELSE (wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - not found.")
ENDIF (wxWidgets_FOUND)
IF (Matlab_FOUND)
  MESSAGE (STATUS "Looking for Matlab - found.")
ELSE (Matlab_FOUND)
  MESSAGE (STATUS "Looking for Matlab - not found.")
ENDIF (Matlab_FOUND)
IF (OCTAVE_FOUND)
  MESSAGE (STATUS "Looking for Octave - found.")
ELSE (OCTAVE_FOUND)
  MESSAGE (STATUS "Looking for Octave - not found.")
ENDIF (OCTAVE_FOUND)

##-----------------------------------------------------------------------------
##  I can't remember why this is necessary.  But at one time I needed 
##  it for compiling on AMD linux 64-bit.
##-----------------------------------------------------------------------------
IF ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  SET (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
ENDIF ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")


##-----------------------------------------------------------------------------
# JAS 10.28.2010
# nvcc automatically passed the flag -malign-double to gcc when compiling .cu
# files.  This can be a problem when structs are shared between .c & .cu files on
# 32-bit machines.  So, we pass -malign-double to gcc for .c files as well so
# that everybody is using the same alignment.  -malign-double is automatically
# passed to gcc for 64-bit architectures.
# 
# GCS 2010-10-30
# -malign-double should only be enabled for 32-bit machines.  gcc 4.1.2 
# gives an error if it is used on x86_64.
##-----------------------------------------------------------------------------
#MESSAGE (STATUS "CMAKE_SIZEOF_VOID_P is ${CMAKE_SIZEOF_VOID_P}")
IF (CMAKE_COMPILER_IS_GNUCC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -malign-double")
    SET (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -malign-double")
ENDIF (CMAKE_COMPILER_IS_GNUCC AND CMAKE_SIZEOF_VOID_P EQUAL 4)

##-----------------------------------------------------------------------------
##  Set default build type to "Release"
##-----------------------------------------------------------------------------
##  Unfortunately the below doesn't work.  It gets
##  reset by the ITK include.
##-----------------------------------------------------------------------------
#IF (NOT CMAKE_BUILD_TYPE)
#  SET (CMAKE_BUILD_TYPE Release CACHE STRING
#      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."	
#      FORCE)
#ENDIF (NOT CMAKE_BUILD_TYPE)

##-----------------------------------------------------------------------------
##  Get plastimatch version from subversion 
##-----------------------------------------------------------------------------
SET (PLASTIMATCH_SVN_VERSION "EXPERIMENTAL")
IF (EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  IF (Subversion_FOUND)
    SET (PLASTIMATCH_EXPERIMENTAL TRUE)
    #Subversion_WC_INFO(${CMAKE_SOURCE_DIR} "svn")
    EXECUTE_PROCESS (
      COMMAND 
      ${Subversion_SVNVERSION_EXECUTABLE} "--no-newline" ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE svn_result
      OUTPUT_VARIABLE svn_version
      ERROR_VARIABLE svn_error
      )
    IF (${svn_result} EQUAL 0)
      MESSAGE(STATUS "Looking for version in svn - found")
      SET (PLASTIMATCH_SVN_VERSION ${svn_version})
    ELSE (${svn_result} EQUAL 0)
      MESSAGE(STATUS "Looking for version in svn - can't connect")
    ENDIF (${svn_result} EQUAL 0)
  ELSE (Subversion_FOUND)
    MESSAGE(STATUS "Looking for version in svn - svn not found")
  ENDIF (Subversion_FOUND)
ELSE (EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  SET (PLASTIMATCH_EXPERIMENTAL FALSE)
ENDIF (EXISTS "${CMAKE_SOURCE_DIR}/.svn")

##-----------------------------------------------------------------------------
##  ITK LIBRARY
##-----------------------------------------------------------------------------
FIND_PACKAGE (ITK)
IF (ITK_FOUND)
  INCLUDE (${ITK_USE_FILE})
  SET (ITK_VERSION 
    "${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR}.${ITK_VERSION_PATCH}")
  IF (NOT ITK_USE_REVIEW)
    MESSAGE (FATAL_ERROR 
      "\nFatal Error. ITK must be compiled with ITK_USE_REVIEW set to ON")
  ENDIF (NOT ITK_USE_REVIEW)
IF (ITK_USE_SYSTEM_GDCM)
    MESSAGE (FATAL_ERROR 
      "\nFatal Error. ITK must be compiled with ITK_USE_SYSTEM_GDCM set to OFF")
  ENDIF (ITK_USE_SYSTEM_GDCM)
ENDIF (ITK_FOUND)

##-----------------------------------------------------------------------------
##  WXWINDOWS LIBRARY
##-----------------------------------------------------------------------------
IF (wxWidgets_FOUND)
  IF (wxWidgets_CXX_FLAGS)
    STRING (REGEX REPLACE "-D" "" wxWidgets_DEFINITIONS 
      "${wxWidgets_DEFINITIONS}")
    SET_SOURCE_FILES_PROPERTIES (mondoshot_main.cpp
      PROPERTIES COMPILE_FLAGS 
      ${wxWidgets_CXX_FLAGS}
      )
  ENDIF (wxWidgets_CXX_FLAGS)
ENDIF (wxWidgets_FOUND)

##-----------------------------------------------------------------------------
##  DCMTK LIBRARY
##-----------------------------------------------------------------------------
SET (DCMTK_HAVE_CONFIG_H FALSE)
IF (UNIX)
  SET (DCMTK_HAVE_CONFIG_H TRUE)
ENDIF (UNIX)

##-----------------------------------------------------------------------------
##  Figure out library dependencies
##-----------------------------------------------------------------------------
SET (GPUIT_LIBRARY_DEPENDENCIES "")

# Required
SET (GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${LIBLBFGS_LIBRARIES}
  bstrlib
  nocedal
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
ENDIF (BROOK_FOUND)

# Optional
# -----------------------------------------------
# JAS 2010.11.19
# New CUDA plugin stuff
# -----------------------------------------------
IF (CUDA_FOUND)
  IF (PLM_USE_CUDA_PLUGIN)      # << Use dynamic loading
    IF (WIN32)
      # Windows
      SET (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        plmcuda
        # shared library: cuda extensions
        # we tell windows to "delay load" this
      )
    ELSE (WIN32)
      # Linux
      SET (GPUIT_LIBRARY_DEPENDENCIES
        ${GPUIT_LIBRARY_DEPENDENCIES}
        # In linux we use dlfcn.h to load plmcula
        # in the code @ runtime.
      )
    ENDIF (WIN32)
  ELSE (PLM_USE_CUDA_PLUGIN)    # << use static linking
    SET (GPUIT_LIBRARY_DEPENDENCIES
      ${GPUIT_LIBRARY_DEPENDENCIES}
      ${CUDA_LIBRARIES}
    )
  ENDIF (PLM_USE_CUDA_PLUGIN)
ENDIF (CUDA_FOUND)

# Optional
IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES 
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${FFTW_LIBRARIES})
ENDIF (FFTW_FOUND)

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
  ENDIF (UNIX)
  # Apparently the below is not needed.
  # LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/libf2c)
ELSE (PLM_USE_F2C)
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
  ENDIF (Fortran_COMPILER_NAME STREQUAL "gfortran" 
    OR Fortran_COMPILER_NAME STREQUAL "f95")
ENDIF (PLM_USE_F2C)

# Optional
IF (NLOPT_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${NLOPT_LIBRARIES}
    )
ENDIF (NLOPT_FOUND)

# Optional
IF (OPENCL_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${OPENCL_LIBRARIES}
    )
ENDIF (OPENCL_FOUND)

# Optional
IF (OPENMP_FOUND)
  SET (GPUIT_LIBRARY_DEPENDENCIES
    ${GPUIT_LIBRARY_DEPENDENCIES}
    ${OPENMP_LIBRARIES}
    )
ENDIF (OPENMP_FOUND)

# Required
SET (GPUIT_LIBRARY_DEPENDENCIES
  ${GPUIT_LIBRARY_DEPENDENCIES}
  ${MATH_LIB}
  )

# GPUIT_LIBRARIES with RBF linear system now needs ITK's svd solver
# NSh 7-17-2010
IF (PLM_MONOLITHIC_LIBRARY)
  SET (GPUIT_LIBRARIES 
    plastimatch1 gpuit ${ITK_LIBRARIES} ${GPUIT_LIBRARY_DEPENDENCIES})
  SET (PLASTIMATCH_LIBS ${GPUIT_LIBRARIES})
ELSE (PLM_MONOLITHIC_LIBRARY)
  SET (GPUIT_LIBRARIES gpuit ${GPUIT_LIBRARY_DEPENDENCIES})
  SET (PLASTIMATCH_LIBS plastimatch1 ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
ENDIF (PLM_MONOLITHIC_LIBRARY)

SET (SLICER_PLUGIN_LIBS ${ITK_LIBRARIES} ${GPUIT_LIBRARY_DEPENDENCIES})

#
# JAS 2010.11.22
# Since FDK is roadmapped to fork from plastimatch(?),
# for now we don't have it use the plmcuda plugin.
# Instead it uses direct linkage to the CUDA runtime
# if build machine has CUDA.
SET (FDK_LIBRARIES ${GPUIT_LIBRARIES})
IF (CUDA_FOUND)
  SET (FDK_LIBRARIES ${FDK_LIBRARIES} ${CUDA_LIBRARIES})
ENDIF (CUDA_FOUND)

##-----------------------------------------------------------------------------
##  BROOK FILES
##-----------------------------------------------------------------------------
IF (BROOK_FOUND)
  BRCC_FILE(fdk_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(fdk_brook.cpp PROPERTIES
    OBJECT_DEPENDS "fdk_brook_kernel.cpp")

  BRCC_FILE(convolution_brook_kernel.br)
  BRCC_FILE(demons_brook_kernel.br)
  SET_SOURCE_FILES_PROPERTIES(demons_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "demons_brook_kernel.cpp")
  SET_SOURCE_FILES_PROPERTIES(convolution_brook_kernel_wrap.cpp PROPERTIES
    OBJECT_DEPENDS "convolution_brook_kernel.cpp")

  BRCC_FILE(toy_kernels.br)
  SET_SOURCE_FILES_PROPERTIES(toy_kernels.cpp PROPERTIES
    OBJECT_DEPENDS "toy_kernels.cpp")
ENDIF (BROOK_FOUND)

##-----------------------------------------------------------------------------
##  GPUIT/PLASTIMATCH SOURCE FILES
##-----------------------------------------------------------------------------
SET (GPUIT_LIBRARY_SRC
  bstring_util.cxx bstring_util.h
  bspline.c bspline.h
  bspline_landmarks.c bspline_landmarks.h
  bspline_macros.h
  bspline_mse_cpu_c.c bspline_mse_cpu_c.h
  bspline_optimize.c bspline_optimize.h
  bspline_optimize_lbfgsb.c bspline_optimize_lbfgsb.h
  bspline_optimize_nlopt.c bspline_optimize_nlopt.h
  bspline_optimize_steepest.c bspline_optimize_steepest.h
  bspline_opts.c bspline_opts.h
  bspline_regularize.c bspline_regularize.h
  delayload.c delayload.h
  demons.c demons.h
  demons_cpu.c
  demons_misc.c demons_misc.h
  dir_list.c dir_list.h
  drr.c drr.h
  drr_trilin.c drr_trilin.h
  file_util.cxx file_util.h
  fwrite_block.c fwrite_block.h
  hnd_io.c hnd_io.h
  logfile.c logfile.h
  math_util.h
  mha_io.c mha_io.h
  rpl_volume.c rpl_volume.h
  pointset.c pointset.h 
  print_and_exit.c print_and_exit.h 
  proj_image.c proj_image.h
  proj_image_dir.c proj_image_dir.h
  proj_matrix.c proj_matrix.h 
  proton_dose.c proton_dose.h 
  ray_trace_exact.c ray_trace_exact.h 
  ray_trace_uniform.c ray_trace_uniform.h 
  string_util.c string_util.h
  threading.c threading.h
  plm_timer.c plm_timer.h
  vf.c vf.h
  vf_stats.c vf_stats.h 
  volume.c volume.h
  volume_limit.c volume_limit.h
  xpm.c xpm.h
  )

IF (BROOK_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    demons_brook.cpp
    demons_brook_kernel_wrap.cpp
    demons_brook_kernel.br
    demons_brook_kernel.h
    convolution_brook_kernel.br
    convolution_brook_kernel_wrap.cpp
    convolution_brook_kernel.h
    )
ENDIF (BROOK_FOUND)

IF (CUDA_FOUND)
  IF (PLM_USE_CUDA_PLUGIN)      # << Dynamic loading
    SET (PLM_CUDA_BSPLINE_SRC
      bspline_cuda.cpp
      bspline_cuda.cu
      bspline_cuda.h
      plm_timer.c         # for bspline_cuda.cpp
      plm_timer.h         # for bspline_cuda.cpp
      logfile.c           # for bspline_cuda.cpp
      logfile.h           # for bspline_cuda.cpp
    )

    SET (PLM_CUDA_DRR_SRC
      drr_cuda.cpp 
      drr_cuda.cu 
      drr_cuda.h 
      drr_cuda_p.h 
    )

    SET (PLM_CUDA_DEMONS_SRC
      demons_cuda.cu
      demons_misc.c
      demons_misc.h
      volume.c            # for demons_cuda.cu
      volume.h            # for demons_cuda.cu
      print_and_exit.c    # for volume.c
      print_and_exit.h    # for volume.c
    )

    SET (PLM_CUDA_COMMON
      cuda_util.cu
      cuda_util.h
      cuda_kernel_util.cu
      cuda_kernel_util.h
      cuda_mem.cu
      cuda_mem.h
    )

    SET (PLM_CUDA_LIBRARY_SRC
      ${PLM_CUDA_BSPLINE_SRC}
      ${PLM_CUDA_DRR_SRC}
      ${PLM_CUDA_DEMONS_SRC}
      ${PLM_CUDA_COMMON}
    )

    CUDA_ADD_LIBRARY(plmcuda SHARED
        ${PLM_CUDA_LIBRARY_SRC}
    )
  ELSE (PLM_USE_CUDA_PLUGIN)        # << Static linking
    SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    bspline_cuda.cpp
    bspline_cuda.cu
    bspline_cuda.h
    cuda_util.cu cuda_util.h
    cuda_kernel_util.cu cuda_kernel_util.h
    cuda_mem.cu cuda_mem.h
    )
  CUDA_COMPILE (BSPLINE_CUDA_WRAPPERS 
    bspline_cuda.cu 
    cuda_util.cu
    cuda_kernel_util.cu
    cuda_mem.cu
    )
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ${BSPLINE_CUDA_WRAPPERS}
    )
  CUDA_COMPILE (DRR_CUDA_WRAPPERS drr_cuda.cu)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ${DRR_CUDA_WRAPPERS}
    drr_cuda.cpp 
    drr_cuda.cu 
    drr_cuda.h 
    drr_cuda_p.h 
    )
  CUDA_COMPILE (DEMONS_CUDA_WRAPPERS demons_cuda.cu)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ${DEMONS_CUDA_WRAPPERS}
    demons_cuda.cu
    )
  ENDIF (PLM_USE_CUDA_PLUGIN)
ENDIF (CUDA_FOUND)

IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ramp_filter.c ramp_filter.h
   )
ENDIF (FFTW_FOUND)

IF (OPENCL_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    autotune_opencl.cxx autotune_opencl.h
    demons_opencl.cxx demons_opencl_p.h demons_opencl.cl
    drr_opencl.cxx drr_opencl.h drr_opencl_p.h drr_opencl.cl
    opencl_probe.cxx opencl_probe.h opencl_probe.cl
    opencl_util.cxx opencl_util.h
    opencl_util_nvidia.cxx opencl_util_nvidia.h
    )
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC demons_opencl.cl)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC drr_opencl.cl)
ENDIF (OPENCL_FOUND)

SET (GPUIT_LIBRARY_SRC
  ${GPUIT_LIBRARY_SRC}
  bspline_optimize_liblbfgs.c bspline_optimize_liblbfgs.h
  )

IF (F2C_LIBRARY)
  SET (F2C_HELPER_LIBRARY_SRC
    main__.c
    )
ENDIF (F2C_LIBRARY)

SET (PLASTIMATCH_LIBRARY_SRC
  astroid_dose.cxx astroid_dose.h
  cxt_io.cxx cxt_io.h 
  cxt_to_mha.cxx cxt_to_mha.h 
  demographics.cxx demographics.h 
  gcs_metric.h gcs_metric.hxx 
  gdcm_dose.cxx gdcm_dose.h
  gdcm_rtss.cxx gdcm_rtss.h
  gdcm_series.cxx gdcm_series.h
  gdcm_series_helper_2.cxx gdcm_series_helper_2.h
  gpuit_bspline.cxx gpuit_bspline.h 
  gpuit_demons.cxx gpuit_demons.h 
  itkAndConstantToImageFilter.h
  itk_crop.cxx itk_crop.h 
  itk_demons.cxx itk_demons.h 
  itk_dicom.cxx itk_dicom.h 
  itk_image.cxx itk_image.h 
  itk_image_cast.cxx itk_image_cast.h 
  itk_image_save.cxx itk_image_save.h 
  itk_image_stats.cxx itk_image_stats.h 
  itk_pointset.cxx itk_pointset.h
  itk_optim.cxx itk_optim.h 
  itk_registration.cxx itk_registration.h 
  itk_tps.cxx itk_tps.h 
  itk_warp.cxx itk_warp.h 
  landmark_warp.cxx landmark_warp.h 
  mask_mha.cxx mask_mha.h 
  mc_dose.cxx mc_dose.h
  plm_api.cxx plm_api.h 
  plm_file_format.cxx plm_file_format.h 
  plm_image.cxx plm_image.h 
  plm_image_convert.cxx plm_image_convert.h 
  plm_image_header.cxx plm_image_header.h 
  plm_image_patient_position.cxx plm_image_patient_position.h 
  plm_image_type.cxx plm_image_type.h 
  plm_int.h
  plm_parms.cxx plm_parms.h 
  plm_register_loadable.cxx plm_register_loadable.h 
  plm_stages.cxx plm_stages.h 
  plm_version.h 
  plm_warp.cxx plm_warp.h 
  rbf_gauss.cxx rbf_gauss.h
  rbf_gcs.cxx rbf_gcs.h
  referenced_dicom_dir.cxx referenced_dicom_dir.h 
  render_polyline.cxx render_polyline.h
  resample_mha.cxx resample_mha.h 
  rtds.cxx rtds.h
  rtds_dicom.cxx rtds_dicom.h
  rtds_warp.cxx rtds_warp.h
  rtss_polyline_set.cxx rtss_polyline_set.h 
  rtss_structure.cxx rtss_structure.h 
  slice_extract.cxx slice_extract.h
  ss_image.cxx ss_image.h
  ss_img_extract.cxx ss_img_extract.h
  ss_list_io.cxx ss_list_io.h
  synthetic_mha.cxx synthetic_mha.h
  xio_ct.cxx xio_ct.h
  xio_dir.cxx xio_dir.h
  xio_dose.cxx xio_dose.h
  xio_io.cxx xio_io.h
  xio_plan.cxx xio_plan.h
  xio_structures.cxx xio_structures.h
  xform.cxx xform.h 
  )

IF (ITK_USE_REVIEW)
  SET (PLASTIMATCH_LIBRARY_SRC
    ${PLASTIMATCH_LIBRARY_SRC}
    cxt_extract.cxx cxt_extract.h)
ENDIF (ITK_USE_REVIEW)

##-----------------------------------------------------------------------------
##  SOURCE FILES (EXECUTABLES)
##-----------------------------------------------------------------------------
SET (BRAGG_CURVE_SRC
  bragg_curve_main.c 
  bragg_curve_opts.c bragg_curve_opts.h
  )
SET (BSPLINE_SRC
  bspline_main.c 
  )
SET (CHECK_GRAD_SRC
  check_grad.c
  check_grad_opts.c check_grad_opts.h 
  )
SET (COMPUTE_DISTANCE_SRC
  compute_distance.cxx compute_distance.h
  closest_point.cxx 
  read_obj.cxx
  )
SET (COMPUTE_MEAN_IMAGE_SRC
  compute_mean_image_main.cxx
  )
SET (CUDA_PROBE_SRC
  cuda_probe_main.c
  cuda_probe.cu
  cuda_probe.h
  )
IF (CUDA_FOUND)
  CUDA_COMPILE(CUDA_PROBE_WRAPPERS cuda_probe.cu)
  SET (CUDA_PROBE_SRC
    ${CUDA_PROBE_SRC}
    ${CUDA_PROBE_WRAPPERS}
    )
ENDIF (CUDA_FOUND)
SET (DEMONS_SRC
  demons_main.c
  demons_opts.c demons_opts.h
  delayload.c delayload.h
  )
SET (DICE_STATS_SRC
  contour_statistics.cxx contour_statistics.h
  dice_statistics.cxx 
  )
SET (DICOM_SERIES_TEST_SRC
  dicom_series_test.cxx)
SET (DICOM_UID_SRC
  dicom_uid_main.cxx
  dicom_uid.cxx
  dicom_uid.h
  )
SET (DISTANCE_MAP_SRC
  distance_map.cxx
  )
SET (DRR_SRC
  drr_main.c 
  drr_opts.c drr_opts.h
  )
SET (EXTRACT_CONTOUR_SRC
  extract_contour.cxx
  )
SET (FINDSCU_SRC
  findscu.cc
  )
SET (FDK_SRC
  fdk_main.c 
  fdk_opts.c fdk_opts.h
  fdk_util.c fdk_util.h
  bowtie_correction.c bowtie_correction.h
  )
IF (BROOK_FOUND)
  SET (FDK_SRC
    ${FDK_SRC}
    fdk_brook.cpp
    fdk_brook_kernel.br
    )
ENDIF (BROOK_FOUND)

IF (CUDA_FOUND)
  IF (PLM_USE_CUDA_PLUGIN)  # << FDK does not use cuda_plugin
    SET (FDK_SRC            #    so compile in stuff it needs
      ${FDK_SRC}            #    that now lives in plugin
      fdk_cuda.cu 
      fdk_cuda_p.h
      fdk_cuda.h
      cuda_util.cu
      cuda_util.h
    )
  ELSE (PLM_USE_CUDA_PLUGIN)   # << Static linking.  Everything
    SET (FDK_SRC               #    links against GPUIT, which
      ${FDK_SRC}               #    includes cuda_utils
      fdk_cuda.cu 
      fdk_cuda_p.h
      fdk_cuda.h
    )
  ENDIF (PLM_USE_CUDA_PLUGIN)
ENDIF (CUDA_FOUND)

IF (OPENCL_FOUND)
  SET (FDK_SRC
    ${FDK_SRC}
    fdk_opencl_old.cxx 
    fdk_opencl.cxx 
    fdk_opencl_p.h
    fdk_opencl.h
    fdk_opencl.cl
    )
  PLM_ADD_OPENCL_FILE (FDK_SRC fdk_opencl.cl)
ENDIF (OPENCL_FOUND)

IF (CUDA_FOUND)
  IF (PLM_USE_CUDA_PLUGIN)
    CUDA_COMPILE (FDK_CUDA_WRAPPERS fdk_cuda.cu cuda_util.cu)
  ELSE (PLM_USE_CUDA_PLUGIN)
    CUDA_COMPILE (FDK_CUDA_WRAPPERS fdk_cuda.cu)
  ENDIF (PLM_USE_CUDA_PLUGIN)

  SET (FDK_SRC
    ${FDK_SRC}
    ${FDK_CUDA_WRAPPERS}
    )
ENDIF (CUDA_FOUND)

SET (HND_TO_PFM_SRC
  hnd_to_pfm.c
  )
SET (ITK_TPS_WARP_SRC
  itk_tps_warp_main.cxx 
  )
SET (MASK_MHA_SRC
  mask_mha_main.cxx
  )
SET (MEX_DRR_SRC
  mex_drr.c
  )
SET (MHA_TO_RTOG_DOSE_SRC
  mha_to_rtog_dose.cxx
  )
SET (MONDOSHOT_SRC
  dicom_uid.cxx dicom_uid.h
  mondoshot_dicom.cpp mondoshot_dicom.h
  mondoshot_main.cpp mondoshot_main.h
  )
SET (MONDOSHOT_RELAY_SRC
  mondoshot_relay_main.cpp
  )
SET (MOVESCU_SRC
  movescu.cc
  )
SET (OPENCL_PROBE_SRC
  opencl_probe_main.c
  )
IF (OPENCL_FOUND)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC opencl_probe.cl)
ENDIF (OPENCL_FOUND)
SET (PATIENT_MASK_SRC
  patient_mask.cxx
  )
SET (PLASTIMATCH_SRC
  pcmd_add.cxx pcmd_add.h
  pcmd_adjust.cxx pcmd_adjust.h
  pcmd_compare.cxx pcmd_compare.h
  pcmd_compose.cxx pcmd_compose.h
  pcmd_crop.cxx pcmd_crop.h
  pcmd_diff.cxx pcmd_diff.h
  pcmd_dvh.cxx pcmd_dvh.h
  pcmd_header.cxx pcmd_header.h
  pcmd_mask.cxx pcmd_mask.h
  pcmd_resample.cxx pcmd_resample.h
  pcmd_segment.cxx pcmd_segment.h
  pcmd_slice.cxx pcmd_slice.h 
  pcmd_stats.cxx pcmd_stats.h
  pcmd_warp_dij.cxx 
  pcmd_warp.cxx pcmd_warp.h 
  pcmd_warp_pointset.cxx 
  plastimatch_main.cxx
  )
SET (PLASTIMATCH_SLICER_SRC
  plastimatch-slicer.cxx
  )
SET (PLASTIMATCH_SLICER_TPS_SRC
  plastimatch-slicer-tps.cxx
  )
SET (PLASTIMATCH_SLICER_DICOMRT_SRC
  plastimatch-slicer-dicomrt.cxx
  )
SET (PLASTIMATCH_QT_SRC
  # pqt_application.cxx pqt_application.h
  pqt_data_source_dialog.cxx pqt_data_source_dialog.h
  pqt_data_source_list_model.cxx pqt_data_source_list_model.h
  pqt_database.cxx pqt_database.h
  pqt_findscu.cxx pqt_findscu.h
  pqt_main.cxx
  pqt_main_window.cxx pqt_main_window.h
  pqt_patient_list_model.cxx pqt_patient_list_model.h
  )
IF (QT4_FOUND AND QT_QTSQL_FOUND)
  QT4_WRAP_CPP (PLASTIMATCH_QT_SRC 
    #    pqt_application.h
    pqt_data_source_dialog.h
    pqt_data_source_list_model.h
    pqt_main_window.h 
    pqt_patient_list_model.h
    )
  QT4_WRAP_UI (PLASTIMATCH_QT_SRC 
    pqt_data_source_dialog.ui
    pqt_main_window.ui
    )
ENDIF (QT4_FOUND AND QT_QTSQL_FOUND)
SET (PROTON_DOSE_SRC
  proton_dose_main.c 
  proton_dose_opts.c proton_dose_opts.h
  )
WRAP_GGO (landmark_warp_GGO_C landmark_warp.ggo)
MESSAGE (STATUS "landmark_warp_GGO_C = ${landmark_warp_GGO_C}")
SET (LANDMARK_WARP_SRC
  landmark_warp_main.cxx
  landmark_warp_args.cxx landmark_warp_args.h
  ${landmark_warp_GGO_C}
  )
SET (RTOG_TO_MHA_SRC
  exchkeys.cxx exchkeys.h
  rtog_to_mha.cxx
  )
SET (SHUFFLE_MHA_SRC
  shuffle_mha_main.cxx
  )
SET (SYNTHETIC_MHA_SRC
  synthetic_mha_main.cxx synthetic_mha_main.h
  )
SET (VF_INVERT_SRC
  vf_invert.cxx
  )
SET (XF_TO_XF_SRC
  xf_to_xf_main.cxx xf_to_xf_main.h
  )
SET (XF_TO_EMPIREFMT_SRC
  xf_to_empirefmt.cxx
  )


CHECK_FUNCTION_EXISTS (getopt_long HAVE_GETOPT_LONG)
IF (NOT HAVE_GETOPT_LONG)
    SET (DICOM_SERIES_TEST_SRC ${DICOM_SERIES_TEST_SRC} getopt_long.c getopt.h)
    SET (MHA_TO_RTOG_DOSE_SRC ${MHA_TO_RTOG_DOSE_SRC} getopt_long.c getopt.h)
    SET (PATIENT_MASK_SRC ${PATIENT_MASK_SRC} getopt_long.c getopt.h)
    SET (PLASTIMATCH_SRC ${PLASTIMATCH_SRC} getopt_long.c getopt.h)
    SET (RTOG_TO_MHA_SRC ${RTOG_TO_MHA_SRC} getopt_long.c getopt.h)
    SET (COMPUTE_MEAN_IMAGE_SRC ${COMPUTE_MEAN_IMAGE_SRC} getopt_long.c getopt.h)
    SET (SYNTHETIC_MHA_SRC ${SYNTHETIC_MHA_SRC} getopt_long.c getopt.h)
    SET (VF_INVERT_SRC ${VF_INVERT_SRC} getopt_long.c getopt.h)
    SET (XF_TO_XF_SRC ${XF_TO_XF_SRC} getopt_long.c getopt.h)
ENDIF (NOT HAVE_GETOPT_LONG)

##-----------------------------------------------------------------------------
##  SOURCE FILE PROPERTIES
##-----------------------------------------------------------------------------
IF (OPENMP_FOUND)
  SET_SOURCE_FILES_PROPERTIES (bspline.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (drr.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (fdk_main.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (openmp_test.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})

ENDIF (OPENMP_FOUND)

IF (SSE2_FOUND)
# JAS 08.19.2010 - Unfortunately, this doesn't work.
#  SET_PROPERTY(
#      SOURCE bspline.c
#      APPEND PROPERTY COMPILE_DEFINITIONS ${SSE2_FLAGS}
#      )

# So, we ask CMake more forcefully to add additional compile flags
    GET_SOURCE_FILE_PROPERTY(OLD_FLAGS bspline.c COMPILE_FLAGS)
    IF (OLD_FLAGS MATCHES "NONE")
        SET(OLD_FLAGS "")
    ENDIF (OLD_FLAGS MATCHES "NONE")
    SET_SOURCE_FILES_PROPERTIES(bspline.c PROPERTIES COMPILE_FLAGS "${OLD_FLAGS} -msse2")
ENDIF (SSE2_FOUND)

##-----------------------------------------------------------------------------
##  CONFIGURE INCLUDE FILES
##-----------------------------------------------------------------------------
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/plm_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/plm_fortran.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_fortran.h)

IF (CMAKE_Fortran_COMPILER_WORKS)
  ##  Ideally I should do this if there is a real fortran compiler.
  ##  create_fortran_c_interface ("" setulb(int i) ${CMAKE_CURRENT_BINARY_DIR}/plm_f2c.h)
ENDIF (CMAKE_Fortran_COMPILER_WORKS)

##-----------------------------------------------------------------------------
##  INCLUDE DIRECTORIES
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES (AFTER ${BSTRLIB_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${LIBLBFGS_INCLUDE_DIR})
INCLUDE_DIRECTORIES (AFTER ${SQLITE_INCLUDE_DIR})

## Don't do the below.  A user-built f2c directory contains ctype.h, which 
## conflicts with the C/C++ header of the same name.
#IF (PLM_USE_F2C)
#  INCLUDE_DIRECTORIES (BEFORE ${F2C_INCLUDE_DIR})
#ENDIF (PLM_USE_F2C)

IF (BROOK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${BROOK_INCLUDE_DIR})
ENDIF (BROOK_FOUND)

IF (CUDA_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${CUDA_INCLUDE_DIRS})
  ##INCLUDE_DIRECTORIES (AFTER ${CUDA_CUT_INCLUDE_DIR})
ENDIF (CUDA_FOUND)

IF (DCMTK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${DCMTK_INCLUDE_DIR})
ENDIF (DCMTK_FOUND)

IF (FFTW_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${FFTW_INCLUDE_DIR})
  LINK_DIRECTORIES (${FFTW_DIR})
ENDIF (FFTW_FOUND)

IF (OPENCL_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${OPENCL_INCLUDE_DIR})
ENDIF (OPENCL_FOUND)

IF (MATLAB_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${MATLAB_INCLUDE_DIRS})
ENDIF (MATLAB_FOUND)

IF (PANTHEIOS_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${STLSOFT_INCLUDE_DIR})
  INCLUDE_DIRECTORIES (AFTER ${PANTHEIOS_INCLUDE_DIR})
ENDIF (PANTHEIOS_FOUND)

IF (QT4_FOUND AND QT_QTSQL_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${QT_QTSQL_INCLUDE_DIR})
ENDIF (QT4_FOUND AND QT_QTSQL_FOUND)


IF (SLICER_FOUND)
  MESSAGE (STATUS "SLICER INCLUDE DIRS: ${Slicer_Libs_INCLUDE_DIRS}")
  MESSAGE (STATUS "SLICER INCLUDE DIRS: ${Slicer_Base_INCLUDE_DIRS}")
  INCLUDE_DIRECTORIES (AFTER
    ${SLICER_INCLUDE_DIRS}
    #${Slicer_Libs_INCLUDE_DIRS}
    #${Slicer_Base_INCLUDE_DIRS}
    )
ENDIF (SLICER_FOUND)

IF (wxWidgets_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${wxWidgets_INCLUDE_DIRS})
ENDIF (wxWidgets_FOUND)

##-----------------------------------------------------------------------------
##  Subdirectories
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (fatm/src)
ADD_SUBDIRECTORY (CUDA_tex_test)

##-----------------------------------------------------------------------------
##  Library targets
##-----------------------------------------------------------------------------
IF (F2C_LIBRARY AND UNIX)
  PLM_ADD_LIBRARY (f2c_helper "${F2C_HELPER_LIBRARY_SRC}" "")
ENDIF (F2C_LIBRARY AND UNIX)

PLM_ADD_LIBRARY (gpuit "${GPUIT_LIBRARY_SRC}" "${GPUIT_LIBRARY_DEPENDENCIES}")

# JAS 2010.11.23
# Because the cuda plugin is loaded at runtime, CMake doesn't know
# that it is needed by everything linked against gpuit.  So, we tell it.
IF (PLM_USE_CUDA_PLUGIN AND CUDA_FOUND)
    ADD_DEPENDENCIES (gpuit plmcuda)
ENDIF (PLM_USE_CUDA_PLUGIN AND CUDA_FOUND)

IF (ITK_FOUND)
  ## Despite claims to the contrary, you are not allowed to have a library
  ## with the same name as the executable.  What happens is the executable 
  ## tries to create a temporary library with the same name.  [MSVC 2005]
  ## That is why the library is called plastimatch1 instead of plastimatch.
  SET (PLASTIMATCH_LIBRARY_DEPENDENCIES ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
  PLM_ADD_LIBRARY (plastimatch1 "${PLASTIMATCH_LIBRARY_SRC}" 
    "${PLASTIMATCH_LIBRARY_DEPENDENCIES}")
ENDIF (ITK_FOUND)

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------

# Consolidate linker flags
SET (GPUIT_LDFLAGS "${OPENMP_LDFLAGS}")
IF (WIN32 AND NOT CYGWIN AND NOT MINGW)
  IF (PLM_USE_CUDA_PLUGIN)
    SET (GPUIT_LDFLAGS "${GPUIT_LDFLAGS} /DELAYLOAD:plmcuda.dll /DELAYLOAD:opencl.dll")
  ELSE (PLM_USE_CUDA_PLUGIN)
    SET (GPUIT_LDFLAGS "${GPUIT_LDFLAGS} /DELAYLOAD:nvcuda.dll /DELAYLOAD:opencl.dll")
  ENDIF (PLM_USE_CUDA_PLUGIN)
ELSE (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET (GPUIT_LDFLAGS "${GPUIT_LDFLAGS} -ldl")
ENDIF (WIN32 AND NOT CYGWIN AND NOT MINGW)
MESSAGE (STATUS "GPUIT_LDFLAGS: ${GPUIT_LDFLAGS}")

# Executable Target Definitions
PLM_ADD_EXECUTABLE (bragg_curve "${BRAGG_CURVE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
PLM_ADD_EXECUTABLE (bspline "${BSPLINE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
PLM_ADD_EXECUTABLE (check_grad "${CHECK_GRAD_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
PLM_ADD_EXECUTABLE (demons "${DEMONS_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
PLM_ADD_EXECUTABLE (drr "${DRR_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
IF (PLM_USE_CUDA_PLUGIN)
  PLM_ADD_EXECUTABLE (fdk "${FDK_SRC}" "${FDK_LIBRARIES}" "${GPUIT_LDFLAGS}")
ELSE (PLM_USE_CUDA_PLUGIN)
  PLM_ADD_EXECUTABLE (fdk "${FDK_SRC}" "${GPUIT_LIBRARIES}" "${GPUIT_LDFLAGS}")
ENDIF (PLM_USE_CUDA_PLUGIN)
PLM_ADD_EXECUTABLE (hnd_to_pfm "${HND_TO_PFM_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
PLM_ADD_EXECUTABLE (mha_to_raw mha_to_raw.c "" "")
PLM_ADD_EXECUTABLE (proton_dose "${PROTON_DOSE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_LDFLAGS}")
PLM_ADD_EXECUTABLE (raw_to_mha raw_to_mha.c "" "")

IF (ITK_FOUND)
  MESSAGE (STATUS "PLASTIMATCH_LIBS: ${PLASTIMATCH_LIBS}")
  PLM_ADD_EXECUTABLE (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  IF (ITK_USE_REVIEW)
    PLM_ADD_EXECUTABLE (compute_mean_image "${COMPUTE_MEAN_IMAGE_SRC}"
      "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  ENDIF (ITK_USE_REVIEW)

  PLM_ADD_EXECUTABLE (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (dicom_series_test "${DICOM_SERIES_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (distance_map "${DISTANCE_MAP_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  IF (ITK_USE_REVIEW)
    PLM_ADD_EXECUTABLE (extract_contour "${EXTRACT_CONTOUR_SRC}"
      "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  ENDIF (ITK_USE_REVIEW)
  PLM_ADD_EXECUTABLE (itk_tps_warp "${ITK_TPS_WARP_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (mask_mha "${MASK_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (merge_vfs merge_vector_fields.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (patient_mask "${PATIENT_MASK_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (plastimatch "${PLASTIMATCH_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  IF (SLICER_FOUND)
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer 
      "${PLASTIMATCH_SLICER_SRC}" "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer-dicomrt
      "${PLASTIMATCH_SLICER_DICOMRT_SRC}" "${PLASTIMATCH_LIBS}" 
      "${GPUIT_LDFLAGS}")
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer-tps 
      "${PLASTIMATCH_SLICER_TPS_SRC}" "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  ENDIF (SLICER_FOUND)
  IF (QT4_FOUND AND QT_QTSQL_FOUND)
    SET (PLASTIMATCH_QT_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    PLM_ADD_EXECUTABLE (plastimatch_qt "${PLASTIMATCH_QT_SRC}"
      "${PLASTIMATCH_QT_LIBRARIES}" "${GPUIT_LDFLAGS}")
  ENDIF (QT4_FOUND AND QT_QTSQL_FOUND)
  PLM_ADD_EXECUTABLE (rtog_to_mha "${RTOG_TO_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (synthetic_mha "${SYNTHETIC_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (landmark_warp "${LANDMARK_WARP_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (union_mask union_mask.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (vf_invert "${VF_INVERT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (xf_to_xf "${XF_TO_XF_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
  PLM_ADD_EXECUTABLE (xf_to_EMPIREFMT "${XF_TO_EMPIREFMT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_LDFLAGS}")
ENDIF (ITK_FOUND)

# executables that require cuda
IF (CUDA_FOUND)
  PLM_ADD_EXECUTABLE (cuda_probe "${CUDA_PROBE_SRC}" "${CUDA_LIBRARIES}" "")
ENDIF (CUDA_FOUND)

# executables that require opencl
IF (OPENCL_FOUND)
  PLM_ADD_EXECUTABLE (opencl_probe 
    "${OPENCL_PROBE_SRC}"  "${GPUIT_LIBRARIES}" "${GPUIT_LDFLAGS}")
ENDIF (OPENCL_FOUND)

# dicom_uid requires dcmtk
IF (DCMTK_FOUND)
  PLM_ADD_EXECUTABLE (dicom_uid "${DICOM_UID_SRC}" "${DCMTK_LIBRARIES}" "")
  PLM_ADD_EXECUTABLE (findscu "${FINDSCU_SRC}" "${DCMTK_LIBRARIES}" "")
  PLM_ADD_EXECUTABLE (movescu "${MOVESCU_SRC}" "${DCMTK_LIBRARIES}" "")
ENDIF (DCMTK_FOUND)

# mondoshot requires WIN32, wx, dcmtk, sqlite3
IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  ADD_EXECUTABLE(mondoshot WIN32 ${MONDOSHOT_SRC})
  TARGET_LINK_LIBRARIES(mondoshot ${wxWidgets_LIBRARIES} ${DCMTK_LIBRARIES}
    ${SQLITE_LIBRARIES})
  IF (NOT BUILD_AGAINST_SLICER3)
    INSTALL(TARGETS mondoshot DESTINATION bin)
  ENDIF (NOT BUILD_AGAINST_SLICER3)
ENDIF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)

# matlab mex files
IF (PLM_BUILD_MEX_TEST)
IF (MATLAB_FOUND)
  MEX_TARGET (mex_drr "${MEX_DRR_SRC}" "${GPUIT_LIBRARIES}" "${GPUIT_LDFLAGS}")
ENDIF (MATLAB_FOUND)
ENDIF (PLM_BUILD_MEX_TEST)

##-----------------------------------------------------------------------------
##  Slicer loadable module (in development)
##-----------------------------------------------------------------------------
IF (SLICER_FOUND AND NOT SLICER_IS_SLICER3 AND PLM_BUILD_SLICER_LOADABLE)

  SET (CMAKE_MODULE_PATH "${Slicer_DIR}/CMake" ${CMAKE_MODULE_PATH})
  #INCLUDE ("${Slicer_CMAKE_DIR}/SlicerQTModuleMacros.cmake")
  #INCLUDE ("${Slicer_CMAKE_DIR}/SlicerParseArgumentsMacro.cmake")
  ADD_SUBDIRECTORY (slicer_loadable)

  IF (COMMENTOUT)
    CONFIGURE_FILE (
      ${CMAKE_CURRENT_SOURCE_DIR}/vtkPlastimatchLoadableConfigure.h.in 
      ${CMAKE_CURRENT_BINARY_DIR}/vtkPlastimatchLoadableConfigure.h
      )
    set (PlastimatchLoadable_SRCS
      vtkPlastimatchLoadableLogic.cxx 
      vtkPlastimatchLoadableGUI.cxx 
      vtkMRMLPlastimatchLoadableNode.cxx 
      )

    include("${VTK_CMAKE_DIR}/vtkWrapTcl.cmake")
    vtk_wrap_tcl3 (PlastimatchLoadable
      PlastimatchLoadable_TCL_SRCS 
      "${PlastimatchLoadable_SRCS}" "")
    MESSAGE (STATUS "TCL SOURCES: ${PlastimatchLoadable_TCL_SRCS}")

    GENERATELM (PlastimatchLoadable_SRCS PlastimatchLoadable.xml)

    PLM_ADD_SLICER_MODULE (PlastimatchLoadable
      "${PlastimatchLoadable_SRCS};${PlastimatchLoadable_TCL_SRCS}" 
      "${PLASTIMATCH_LIBS}")

    ADD_LIBRARY (PlastimatchLoadable
      ${PlastimatchLoadable_SRCS}
      ${PlastimatchLoadable_TCL_SRCS})
    SLICER3_SET_MODULES_OUTPUT_PATH (PlastimatchLoadable)
    TARGET_LINK_LIBRARIES (PlastimatchLoadable
	${Slicer_Libs_LIBRARIES}
	${Slicer_Base_LIBRARIES}
	${KWWidgets_LIBRARIES}
	${ITK_LIBRARIES})
      #      ${PLASTIMATCH_LIBS})
    SLICER3_INSTALL_MODULES (PlastimatchLoadable)
  ENDIF (COMMENTOUT)

ENDIF (SLICER_FOUND AND NOT SLICER_IS_SLICER3 AND PLM_BUILD_SLICER_LOADABLE)

##-----------------------------------------------------------------------------
##  Additional install files
##-----------------------------------------------------------------------------
IF (ITK_FOUND)
  IF (EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")
    INSTALL(FILES "${ITK_DIR}/bin/release/ITKCommon.dll" DESTINATION bin)
  ENDIF (EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")

  INSTALL(FILES 
    "${CMAKE_SOURCE_DIR}/cttools/dicomrt_to_cxt.pl" 
    "${CMAKE_SOURCE_DIR}/cttools/dcm_image_uids.pl" 
    "${CMAKE_SOURCE_DIR}/cttools/cxt_apply_uids.pl"
    "${CMAKE_SOURCE_DIR}/cttools/parse_cxt.pl"
    DESTINATION bin)
ENDIF (ITK_FOUND)

IF (CUDA_FOUND)
  IF (UNIX)
    # Hard code to cuda 2.2.  Note, we copy it first, because otherwise 
    # CMake 2.6 will install a broken symbolic link
    SET (CUDART_FILE_SRC "${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.so.2.2")
    SET (CUDART_FILE "${CMAKE_BINARY_DIR}/libcudart.so.2")
    IF (EXISTS "${CUDART_FILE_SRC}")
      EXECUTE_PROCESS (COMMAND ${CMAKE_COMMAND} "-E" "copy" 
	"${CUDART_FILE_SRC}" "${CUDART_FILE}")
    ENDIF (EXISTS "${CUDART_FILE_SRC}")
  ELSE (UNIX)
    SET (CUDART_FILE "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart.dll")
  ENDIF (UNIX)

  IF (EXISTS "${CUDART_FILE}")
    INSTALL (FILES "${CUDART_FILE}" DESTINATION bin)
    IF (BUILD_AGAINST_SLICER3)
      INSTALL (FILES "${CUDART_FILE}" DESTINATION lib/Slicer3/Plugins)
    ENDIF (BUILD_AGAINST_SLICER3)
  ENDIF (EXISTS "${CUDART_FILE}")

ENDIF (CUDA_FOUND)

IF (NOT BUILD_AGAINST_SLICER3)
  INSTALL (FILES 
    COPYRIGHT.TXT
    INSTALL.TXT
    LICENSE.TXT
    README.TXT
    README.BROOK.TXT
    REGISTRATION.TXT
    DESTINATION doc)
ENDIF (NOT BUILD_AGAINST_SLICER3)

##-----------------------------------------------------------------------------
##  Simple test files
##-----------------------------------------------------------------------------
# Test executable -- cuda
IF (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)
  CUDA_COMPILE (CUDA_TEST_WRAPPERS cuda_test.cu)
  SET (CUDA_TEST_SRC ${CUDA_TEST_WRAPPERS} cuda_test.cu)
  MESSAGE (STATUS "CUDA_TEST_SRC: ${CUDA_TEST_SRC}")
  PLM_ADD_EXECUTABLE (cuda_test "${CUDA_TEST_SRC}" "${CUDA_LIBRARIES}" "")
  SET_TARGET_PROPERTIES (cuda_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)

# Test executable -- fann
IF (PLM_BUILD_FANN_TEST AND FANN_FOUND)
  SET (FANN_TEST_SRC fann_test.c)
  SET (FANN_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${FANN_LIBRARIES})
  PLM_ADD_EXECUTABLE (fann_test "${FANN_TEST_SRC}" "${FANN_TEST_LIBS}" 
    "${OPENMP_LDFLAGS}")
ENDIF (PLM_BUILD_FANN_TEST AND FANN_FOUND)

# Test executable -- mex
IF (PLM_BUILD_MEX_TEST)
  IF (OCTAVE_FOUND)
    ADD_CUSTOM_COMMAND (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test.mex"
      COMMAND ${OCTAVE_MKOCTFILE} --mex 
      -o "${CMAKE_BINARY_DIR}/mex_test.mex" 
      "${CMAKE_SOURCE_DIR}/mex_test.c"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    ADD_CUSTOM_TARGET (oct_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test.mex")
  ENDIF (OCTAVE_FOUND)
  IF (MATLAB_FOUND)
    FILE (WRITE "${CMAKE_BINARY_DIR}/compile_mex_test.m"
      "mex \"${CMAKE_SOURCE_DIR}/mex_test.c\";exit;\n")
    FILE (WRITE "${CMAKE_BINARY_DIR}/mex_test.cmake"
      "EXECUTE_PROCESS (COMMAND ${MATLAB_EXE} -nosplash -nodesktop -nojvm
       -r compile_mex_test
       RESULT_VARIABLE RESULT
       OUTPUT_VARIABLE STDOUT
       ERROR_VARIABLE STDERR)\n")
    ADD_CUSTOM_COMMAND (
      OUTPUT "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}"
      COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/mex_test.cmake"
      DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
    ADD_CUSTOM_TARGET (mat_mex_test
      DEPENDS "${CMAKE_BINARY_DIR}/mex_test${MATLAB_MEXEXT}")
    TARGET_LINK_LIBRARIES (mex_test ${MATLAB_LIBRARIES})
  ENDIF (MATLAB_FOUND)
ENDIF (PLM_BUILD_MEX_TEST)

# Test executable -- nlopt
IF (PLM_BUILD_NLOPT_TEST AND NLOPT_FOUND)
  SET (NLOPT_TEST_SRC nlopt_test.c)
  SET (NLOPT_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${NLOPT_LIBRARIES})
  PLM_ADD_EXECUTABLE (nlopt_test "${NLOPT_TEST_SRC}" "${NLOPT_TEST_LIBS}" 
    "${OPENMP_LDFLAGS}")
  SET_TARGET_PROPERTIES (nlopt_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF (PLM_BUILD_NLOPT_TEST AND NLOPT_FOUND)

# Test executable -- opencl
IF (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)
  SET (OPENCL_TEST_SRC opencl_test.cxx opencl_test.h opencl_test.cl
    "${CMAKE_BINARY_DIR}/opencl_test.cl"
    )
  SET (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${PLASTIMATCH_LIBS})
  #SET (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${GPUIT_LIBRARIES})
  PLM_ADD_EXECUTABLE (opencl_test "${OPENCL_TEST_SRC}" 
    "${OPENCL_TEST_LIBS}" "")
  #SET_TARGET_PROPERTIES (opencl_test PROPERTIES LINKER_LANGUAGE CXX)

  ## I don't yet know how to bundle the .cl file within the executable.
  ## Therefore, copy the .cl into binary directory.
  ADD_CUSTOM_COMMAND (
    OUTPUT "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_SOURCE_DIR}/opencl_test.cl" 
    "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    DEPENDS "${CMAKE_SOURCE_DIR}/opencl_test.cl")
ENDIF (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)

# Test executable -- openmp
IF (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)
  SET (OPENMP_TEST_SRC openmp_test.c)
  PLM_ADD_EXECUTABLE (openmp_test "${OPENMP_TEST_SRC}" "${GPUIT_LIBRARIES}" 
    "${OPENMP_LDFLAGS}")
ENDIF (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)

# Test executable -- qt
IF (PLM_BUILD_QT_TEST AND QT4_FOUND)
  SET (QT_TEST_SRC qt_test.cxx)
  PLM_ADD_EXECUTABLE (qt_test "${QT_TEST_SRC}" "${QT_LIBRARIES}" "")
ENDIF (PLM_BUILD_QT_TEST AND QT4_FOUND)

##-----------------------------------------------------------------------------
##  Regression testing
##  Rationale:
##    1) Tests need to be launched from a script, because 
##       CTest is not able to add the ITK DLL directory 
##       to the path.
##    2) Prefer use of CMAKE as scripting language 
##       to avoid requirement of outside package (e.g. python)
##  Notes/Issues:
##    "Release" is hard coded.  There is (apparently) 
##    no easy way to avoid this (i.e. no good suggestions 
##    on CMake/CTest email list.
##-----------------------------------------------------------------------------
# Set the "hack" variables to find windows binaries (under Release subdir)
IF (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET (PLM_PLASTIMATCH_PATH 
    ${CMAKE_CURRENT_BINARY_DIR}/Release)
  SET (PLM_PLASTIMATCH_TESTING_PATH 
    ${CMAKE_CURRENT_BINARY_DIR}/Testing/Release)
  SET (PLM_FFTW_PATH ${FFTW_DIR})
  SET (PLM_ITK_LIBRARY_PATH_HACK 
    -DITK_LIBRARY_PATH=${ITK_LIBRARY_DIRS}/Release)
  SET (PLM_PLASTIMATCH_PATH_HACK 
     -DPLM_PLASTIMATCH_PATH=${PLM_PLASTIMATCH_PATH})
  SET (PLM_FFTW_PATH_HACK 
     -DPLM_FFTW_PATH=${PLM_FFTW_PATH})
ELSE (WIN32 AND NOT CYGWIN AND NOT MINGW)
  SET (PLM_PLASTIMATCH_PATH ${CMAKE_CURRENT_BINARY_DIR})
  SET (PLM_PLASTIMATCH_TESTING_PATH ${CMAKE_CURRENT_BINARY_DIR}/Testing)
  SET (PLM_FFTW_PATH "${FFTW_DIR}")
  SET (PLM_ITK_LIBRARY_PATH_HACK "")
  SET (PLM_PLASTIMATCH_PATH_HACK "")
  SET (PLM_FFTW_PATH_HACK "")
ENDIF (WIN32 AND NOT CYGWIN AND NOT MINGW)

MACRO (PLM_ADD_TEST TESTNAME WORKING_DIR PLM_TEST_COMMAND PARMS)
  ADD_TEST (${TESTNAME} ${CMAKE_COMMAND} 
    -DTESTNAME=${TESTNAME}
    -DWORKING_DIR=${WORKING_DIR}
    ${PLM_ITK_LIBRARY_PATH_HACK}
    ${PLM_PLASTIMATCH_PATH_HACK}
    ${PLM_FFTW_PATH_HACK}
    -DPLM_TEST_COMMAND=${PLM_TEST_COMMAND}
    -DPLM_TESTING_SOURCE_DIR=${PLM_TESTING_SOURCE_DIR}
    -DPLM_TESTING_BUILD_DIR=${PLM_TESTING_BUILD_DIR}
    "-DPARMS=${PARMS}"
    -P ${CMAKE_SOURCE_DIR}/RUN_CTEST.cmake
    )
ENDMACRO (PLM_ADD_TEST)

# Figure out which tests to ignore
# Note: we need two copies, because unix users might run "make test" 
# in either directory.  MSVC uses the one in ${CMAKE_BINARY_DIR}.
CONFIGURE_FILE (
  ${PLM_TESTING_SOURCE_DIR}/CTestCustom.cmake.in
  ${CMAKE_BINARY_DIR}/CTestCustom.cmake
  @ONLY)
CONFIGURE_FILE (
  ${PLM_TESTING_SOURCE_DIR}/CTestCustom.cmake.in
  ${CMAKE_BINARY_DIR}/Testing/CTestCustom.cmake
  @ONLY)

IF (PLM_BUILD_TESTING)
  ENABLE_TESTING ()
  ADD_SUBDIRECTORY (Testing)

  # Copy the lconv script
  CONFIGURE_FILE (
    "${CMAKE_SOURCE_DIR}/run_lcov.sh" 
    "${CMAKE_BINARY_DIR}/run_lcov.sh"
    COPYONLY)

ENDIF (PLM_BUILD_TESTING)

##-----------------------------------------------------------------------------
##  CPACK stuff
##  Note: "make package_source" to create tarball
##-----------------------------------------------------------------------------
SET (CPACK_GENERATOR "ZIP")
SET (CPACK_SOURCE_GENERATOR "TBZ2")
SET (CPACK_PACKAGE_VERSION_MAJOR "1")
SET (CPACK_PACKAGE_VERSION_MINOR "4-beta")
SET (CPACK_PACKAGE_VERSION_PATCH ${PLASTIMATCH_SVN_VERSION})
SET (CPACK_SOURCE_IGNORE_FILES
  "/.svn"
  "~$"
)
INCLUDE (CPack)
