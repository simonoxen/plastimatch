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
  SET (CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
ENDIF ()

## My modules override CMake defaults
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

##-----------------------------------------------------------------------------
##  Plastimatch configuration variables
##-----------------------------------------------------------------------------
# Building little test programs
OPTION (PLM_BUILD_CUDA_TEST "Build CUDA test program" OFF)
OPTION (PLM_BUILD_DLIB_TEST "Build dlib test program" OFF)
OPTION (PLM_BUILD_FANN_TEST "Build fann test program" OFF)
OPTION (PLM_BUILD_MEX_TEST "Build mex test program" OFF)
OPTION (PLM_BUILD_NLOPT_TEST "Build NLopt test program" OFF)
OPTION (PLM_BUILD_OPENCL_TEST "Build OpenCL test program" OFF)
OPTION (PLM_BUILD_OPENMP_TEST "Build OpenMP test program" OFF)
OPTION (PLM_BUILD_QT_TEST "Build Qt test program" OFF)
OPTION (PLM_BUILD_TCLAP_TEST "Build tclap test program" OFF)

# Software options
OPTION (PLM_USE_4D_SS_IMAGE 
  "Save structure sets as 4D ITK UCHAR images (in development)" OFF)
OPTION (PLM_SLICER_HARDCODED_FILENAME 
  "Use hardcoded filename instead of tmpfile()" OFF)

# Compile and link options
OPTION (BUILD_SHARED_LIBS 
  "Build plastimatch as shared library" OFF)
OPTION (PLM_BUILD_RTK 
  "Build RTK" OFF)
OPTION (PLM_BUILD_SLICER_LOADABLE 
  "Build slicer loadable module" OFF)
OPTION (PLM_PREFER_F2C 
  "Prefer using f2c over native fortran compiler" OFF)
OPTION (PLM_PREFER_SYSTEM_F2C 
  "Prefer the system f2c over the included f2c" ON)

# Installer Options
OPTION (PLM_INSTALL_RPATH 
  "Add full RPATH to install" ON)
OPTION (PLM_OLD_INSTALL_TARGETS 
  "Include old install targets such as static link libraries" OFF)
OPTION (PLM_PACKAGE_32BIT 
  "Set this when building 32-bit packages on a 64-bit machine" OFF)

# Regression testing
OPTION (PLM_BUILD_TESTING 
  "Enable regression testing" ON)
OPTION (PLM_REDUCED_TESTS 
  "Run fewer tests (used when debugging tests)" OFF)
OPTION (PLM_TEST_BSPLINE_FLAVORS 
  "Regression tests for seldom used b-spline implementations" OFF)
OPTION (PLM_TEST_DICOMRT_AW
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
OPTION (PLM_TEST_DICOMRT_PINNACLE 
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
OPTION (PLM_TEST_DICOMRT_XIO 
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
OPTION (PLM_TEST_FDK_VARIAN 
  "Regression tests for fdk with Varian data (downloads extra test data)" OFF)
OPTION (PLM_TEST_XIO 
  "Regression tests for xio file exchange (downloads extra test data)" OFF)

##-----------------------------------------------------------------------------
##  Detect 32-bit or 64-bit machine
##-----------------------------------------------------------------------------
IF (NOT APPLE)
  IF (CMAKE_SIZEOF_VOID_P EQUAL 4)
    SET (HAVE_32_BIT TRUE)
    SET (HAVE_64_BIT FALSE)
    MESSAGE (STATUS "Machine is 32-bit")
  ELSE ()
    SET (HAVE_32_BIT FALSE)
    SET (HAVE_64_BIT TRUE)
    MESSAGE (STATUS "Machine is 64-bit")
  ENDIF ()
ENDIF ()

##-----------------------------------------------------------------------------
##  Extra paths
##-----------------------------------------------------------------------------
SET (PLM_TESTING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Testing")
SET (PLM_TESTING_DATA_DIR "${CMAKE_SOURCE_DIR}/Testing/Data")
SET (PLM_TESTING_DOWNLOAD_DATA_DIR "${CMAKE_BINARY_DIR}/TestData")
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

## JAS 2011.01.24
## I have commented out the INSTALL for the PLM_ADD_LIBRARY
## macro since it was only serving to include static link
## libraries in the CPack generated packages.
## Namely: libplastimatch1.a, libgpuit.a, & libf2c_helper.a
##
MACRO (PLM_ADD_LIBRARY 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  IF (BUILD_AGAINST_SLICER3)
    ADD_LIBRARY (${TARGET_NAME} STATIC ${TARGET_SRC})
    SLICER3_SET_PLUGINS_OUTPUT_PATH (${TARGET_NAME})
  ELSE (BUILD_AGAINST_SLICER3)
    ADD_LIBRARY (${TARGET_NAME} ${TARGET_SRC})
    ##
    IF (PLM_OLD_INSTALL_TARGETS)
      INSTALL (TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
    ENDIF (PLM_OLD_INSTALL_TARGETS)
    ##
  ENDIF (BUILD_AGAINST_SLICER3)
  TARGET_LINK_LIBRARIES (${TARGET_NAME} ${TARGET_LIBS})
  IF (NOT ${TARGET_LDFLAGS} STREQUAL "")
    SET_TARGET_PROPERTIES(${TARGET_NAME} 
      PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
  ENDIF (NOT ${TARGET_LDFLAGS} STREQUAL "")
ENDMACRO (PLM_ADD_LIBRARY)
    
MACRO (PLM_ADD_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS TARGET_INSTALL)

  IF (NOT BUILD_AGAINST_SLICER3)
    ADD_EXECUTABLE(${TARGET_NAME} ${TARGET_SRC})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} ${TARGET_LIBS})
    IF (NOT ${TARGET_LDFLAGS} STREQUAL "")
      SET_TARGET_PROPERTIES(${TARGET_NAME} 
	PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
    ENDIF (NOT ${TARGET_LDFLAGS} STREQUAL "")
    # CXX linkage required for nlopt
    SET_TARGET_PROPERTIES (${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    IF (${TARGET_INSTALL})
      INSTALL(TARGETS ${TARGET_NAME} DESTINATION bin)
    ENDIF (${TARGET_INSTALL})
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
  ENDIF ()
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
#FIND_PACKAGE (Dlib)
FIND_PACKAGE (Etags)
FIND_PACKAGE (Fann)
FIND_PACKAGE (FFTW)
FIND_PACKAGE (Gengetopt)
FIND_PACKAGE (Liblbfgs)
FIND_PACKAGE (NLopt)
IF (PLM_BUILD_MEX_TEST)
  FIND_PACKAGE (Matlab)
ENDIF ()
FIND_PACKAGE (Octave)
FIND_PACKAGE (OpenCL)
FIND_PACKAGE (OpenGL)
FIND_PACKAGE (OpenMP)
FIND_PACKAGE (Qt4 4.6.2 COMPONENTS QtCore QtGui OPTIONAL)
IF (PLM_BUILD_RTK)
  FIND_PACKAGE (RTK)
ENDIF ()
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
ELSE ()
  MESSAGE (STATUS "Looking for Qt4 - not found")
ENDIF ()

##-----------------------------------------------------------------------------
##  Use local libsbfgs if not found
##-----------------------------------------------------------------------------
IF (NOT LIBLBFGS_FOUND)
  ADD_SUBDIRECTORY (libs/liblbfgs-1.9)
  SET (LIBLBFGS_INCLUDE_DIR libs/liblbfgs-1.9/include)
  SET (LIBLBFGS_LIBRARIES lbfgs)
  LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/liblbfgs-1.9)
  SET (LIBLBFGS_FOUND TRUE)
ENDIF ()

##-----------------------------------------------------------------------------
##  Use local sqlite if not found
##-----------------------------------------------------------------------------
IF (NOT SQLITE_FOUND)
  ADD_SUBDIRECTORY (libs/sqlite-3.6.21)
  SET (SQLITE_INCLUDE_DIR libs/sqlite-3.6.21)
  SET (SQLITE_LIBRARIES sqlite3)
ENDIF ()

##-----------------------------------------------------------------------------
##  Only use local bstrlib
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (libs/bstrlib-05122010)
SET (BSTRLIB_INCLUDE_DIR libs/bstrlib-05122010)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/bstrlib-05122010)

##-----------------------------------------------------------------------------
##  Only use local dlib
##-----------------------------------------------------------------------------
SET (DLIB_INCLUDE_DIR libs/dlib-17.34)

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
  # If CUDA is found, use GPU Plugins
  OPTION (PLM_USE_GPU_PLUGINS 
    "Build CUDA support as a runtime loaded shared library" ON)
ELSE (CUDA_FOUND)
  # If CUDA is not found, don't use GPU Plugins and hide the option
  OPTION (PLM_USE_GPU_PLUGINS 
    "CUDA is needed to set this option" OFF)
  MARK_AS_ADVANCED(PLM_USE_GPU_PLUGINS)
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
  ENDIF ()

  # Set reasonable default install prefix and output paths
  # (after setting Slicer3_DIR, delete CMAKE_INSTALL_PREFIX and re-configure)
  slicer3_set_default_install_prefix_for_external_projects ()
  
  #slicer3_enable_plugins_testing()

  # Slicer seems to require a shared library build
  #SET (PLM_BUILD_SHARED_LIBS ON)
ELSE (SLICER_FOUND)
  MESSAGE (STATUS "Looking for Slicer3 - not found")
ENDIF ()

##-----------------------------------------------------------------------------
##  Special linking instructions on unix
##  http://www.cmake.org/Wiki/CMake_RPATH_handling
##-----------------------------------------------------------------------------
IF (PLM_INSTALL_RPATH)
  # use, i.e. don't skip the full RPATH for the build tree
  SET (CMAKE_SKIP_BUILD_RPATH  FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

  # the RPATH to be used when installing
  SET (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
ENDIF ()

##-----------------------------------------------------------------------------
##  User feedback on which languages and libraries were found
##-----------------------------------------------------------------------------
IF (wxWidgets_FOUND)
  MESSAGE (STATUS "Looking for wxWidgets - found.")
ELSE ()
  MESSAGE (STATUS "Looking for wxWidgets - not found.")
ENDIF ()
IF (Matlab_FOUND)
  MESSAGE (STATUS "Looking for Matlab - found.")
ELSE ()
  MESSAGE (STATUS "Looking for Matlab - not found.")
ENDIF ()
IF (OCTAVE_FOUND)
  MESSAGE (STATUS "Looking for Octave - found.")
ELSE ()
  MESSAGE (STATUS "Looking for Octave - not found.")
ENDIF ()

##-----------------------------------------------------------------------------
##  I can't remember why this is necessary.  But at one time I needed 
##  it for compiling on AMD linux 64-bit.
##-----------------------------------------------------------------------------
IF ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  SET (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
ENDIF ()

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
IF (CMAKE_COMPILER_IS_GNUCC AND HAVE_32_BIT)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -malign-double")
    SET (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -malign-double")
ENDIF ()

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
    ELSE ()
      MESSAGE(STATUS "Looking for version in svn - can't connect")
    ENDIF ()
  ELSE ()
    MESSAGE(STATUS "Looking for version in svn - svn not found")
  ENDIF ()
ELSE ()
  SET (PLASTIMATCH_EXPERIMENTAL FALSE)
ENDIF ()

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
  ENDIF ()
IF (ITK_USE_SYSTEM_GDCM)
    MESSAGE (FATAL_ERROR 
      "\nFatal Error. ITK must be compiled with ITK_USE_SYSTEM_GDCM set to OFF")
  ENDIF ()
ENDIF ()

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
  ENDIF ()
ENDIF ()

##-----------------------------------------------------------------------------
##  DCMTK LIBRARY
##-----------------------------------------------------------------------------
SET (DCMTK_HAVE_CONFIG_H FALSE)
IF (UNIX)
  SET (DCMTK_HAVE_CONFIG_H TRUE)
ENDIF ()

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

CHECK_FUNCTION_EXISTS (getopt_long HAVE_GETOPT_LONG)
IF (NOT HAVE_GETOPT_LONG)
  ADD_SUBDIRECTORY (libs/getopt)
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
ENDIF (NLOPT_FOUND)

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
ENDIF ()

##-----------------------------------------------------------------------------
##  PLMSYS/GPUIT/PLASTIMATCH LIBRARY SOURCE FILES
##-----------------------------------------------------------------------------
SET (PLMSYS_LIBRARY_SRC
  file_util.cxx file_util.h
  logfile.c logfile.h
  plm_timer.c plm_timer.h
  print_and_exit.c print_and_exit.h 
  )

SET (GPUIT_LIBRARY_SRC
  bowtie_correction.c bowtie_correction.h
  bstring_util.cxx bstring_util.h
  bspline.c bspline.h
  bspline_landmarks.c bspline_landmarks.h
  bspline_macros.h
  bspline_mse_cpu_c.c bspline_mse_cpu_c.h
  bspline_optimize.c bspline_optimize.h
  bspline_optimize_liblbfgs.c bspline_optimize_liblbfgs.h
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
  fdk.c fdk.h
  fwrite_block.c fwrite_block.h
  hnd_io.c hnd_io.h
  math_util.h
  mha_io.c mha_io.h
  rpl_volume.c rpl_volume.h
  pointset.c pointset.h 
  proj_image.c proj_image.h
  proj_image_dir.c proj_image_dir.h
  proj_matrix.c proj_matrix.h 
  proton_dose.c proton_dose.h 
  ray_trace_exact.c ray_trace_exact.h 
  ray_trace_uniform.c ray_trace_uniform.h 
  string_util.c string_util.h
  threading.c threading.h
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
ENDIF ()

IF (CUDA_FOUND)
  SET (PLMCUDA_LIBRARY_SRC
    bspline_cuda.cpp bspline_cuda.cu bspline_cuda.h
    cuda_kernel_util.cu cuda_kernel_util.h
    cuda_mem.cu cuda_mem.h
    cuda_util.cu cuda_util.h
    demons_cuda.cu demons_cuda.h
    demons_misc.c demons_misc.h
    drr_cuda.cpp drr_cuda.cu drr_cuda.h drr_cuda_p.h 
    fdk_cuda.cpp fdk_cuda.cu fdk_cuda_p.h fdk_cuda.h
    )

  IF (PLM_USE_GPU_PLUGINS)      # << Dynamic loading
    SET (PLMCUDA_LIBRARY_SRC
      ${PLMCUDA_LIBRARY_SRC}
      volume.c            # for demons_cuda.cu
      volume.h            # for demons_cuda.cu
    )
    SET (PLMCUDA_LIBRARY_DEPENDENCIES
      plmsys
    )
    CUDA_ADD_LIBRARY (plmcuda SHARED
        ${PLMCUDA_LIBRARY_SRC}
    )
    TARGET_LINK_LIBRARIES (plmcuda
        ${PLMCUDA_LIBRARY_DEPENDENCIES}
    )
    INSTALL (TARGETS plmcuda
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )
  ELSE ()        # << Static linking
    SET (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${PLMCUDA_LIBRARY_SRC}
      )
    CUDA_COMPILE (CUDA_WRAPPERS 
      bspline_cuda.cu 
      cuda_util.cu
      cuda_kernel_util.cu
      cuda_mem.cu
      demons_cuda.cu
      drr_cuda.cu
      fdk_cuda.cu
      )
    SET (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${CUDA_WRAPPERS}
      )
  ENDIF ()
ENDIF ()

IF (FFTW_FOUND)
  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    ramp_filter.c ramp_filter.h
   )
ENDIF ()

IF (OPENCL_FOUND)
  ## Define OpenCL Plugin (plmopencl)
  SET (PLMOPENCL_LIBRARY_SRC
    autotune_opencl.cxx autotune_opencl.h
    opencl_probe.cxx opencl_probe.h opencl_probe.cl
    opencl_util.cxx opencl_util.h
    opencl_util_nvidia.cxx opencl_util_nvidia.h
    )

  IF (PLM_USE_GPU_PLUGINS)          # << Dynamic linking

    SET (PLMOPENCL_LIBRARY_SRC
      ${PLMOPENCL_LIBRARY_SRC}
      #volume.c            # for demons_opencl.cu
      #volume.h            # for demons_opencl.cu
    )

    SET (PLMOPENCL_LIBRARY_DEPENDENCIES
      plmsys
    )

    # OpenCL Plugin is ALWAYS shared!
    ADD_LIBRARY (plmopencl SHARED
        ${PLMOPENCL_LIBRARY_SRC}
    )

    # Be sure to link against libOpenCL and other depends
    TARGET_LINK_LIBRARIES (plmopencl
        ${OPENCL_LIBRARIES}
        ${PLMOPENCL_LIBRARY_DEPENDENCIES}
    )

    INSTALL (TARGETS plmopencl
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )

  ELSE ()        # << Static linking

    SET (GPUIT_LIBRARY_SRC
      ${GPUIT_LIBRARY_SRC}
      ${PLMOPENCL_LIBRARY_SRC}
    )

  ENDIF ()

  SET (GPUIT_LIBRARY_SRC
    ${GPUIT_LIBRARY_SRC}
    demons_opencl.cxx
    demons_opencl_p.h
    demons_opencl.cl
    drr_opencl.cxx
    drr_opencl_p.h
    drr_opencl.h
    drr_opencl.cl
    fdk_opencl.cxx 
    fdk_opencl_p.h
    fdk_opencl.h
    fdk_opencl.cl
    )
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC demons_opencl.cl)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC drr_opencl.cl)
  PLM_ADD_OPENCL_FILE (GPUIT_LIBRARY_SRC fdk_opencl.cl)

ENDIF ()

IF (F2C_LIBRARY)
  SET (F2C_HELPER_LIBRARY_SRC
    main__.c
    )
ENDIF ()

SET (PLASTIMATCH_LIBRARY_SRC
  astroid_dose.cxx astroid_dose.h
  cxt_extract.cxx cxt_extract.h
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
  plm_drr_api.cxx plm_drr_api.h 
  plm_file_format.cxx plm_file_format.h 
  plm_image.cxx plm_image.h 
  plm_image_convert.cxx plm_image_convert.h 
  plm_image_header.cxx plm_image_header.h 
  plm_image_patient_position.cxx plm_image_patient_position.h 
  plm_image_type.cxx plm_image_type.h 
  plm_int.h
  plm_parms.cxx plm_parms.h 
  plm_register_loadable.cxx plm_register_loadable.h 
  plm_registration_api.cxx plm_registration_api.h 
  plm_stages.cxx plm_stages.h 
  plm_version.h 
  plm_warp.cxx plm_warp.h 
  rbf_gauss.cxx rbf_gauss.h
  rbf_wendland.cxx rbf_wendland.h
  rbf_gcs.cxx rbf_gcs.h
  referenced_dicom_dir.cxx referenced_dicom_dir.h 
  render_polyline.cxx render_polyline.h
  resample_mha.cxx resample_mha.h 
  rtds.cxx rtds.h
  rtds_dicom.cxx rtds_dicom.h
  rtds_warp.cxx rtds_warp.h
  rtss_polyline_set.cxx rtss_polyline_set.h 
  rtss_structure.cxx rtss_structure.h 
  segment_body.cxx segment_body.h
  simplify_points.cxx simplify_points.h
  slice_extract.cxx slice_extract.h
  ss_image.cxx ss_image.h
  ss_img_extract.cxx ss_img_extract.h
  ss_list_io.cxx ss_list_io.h
  synthetic_mha.cxx synthetic_mha.h
  thumbnail.cxx thumbnail.h
  xio_ct.cxx xio_ct.h
  xio_dir.cxx xio_dir.h
  xio_dose.cxx xio_dose.h
  xio_io.cxx xio_io.h
  xio_plan.cxx xio_plan.h
  xio_structures.cxx xio_structures.h
  xform.cxx xform.h 
  xform_point.cxx xform_point.h 
  )

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
ENDIF ()
SET (DEMONS_SRC
  demons_main.c
  demons_opts.c demons_opts.h
  )
SET (DICE_STATS_SRC
  contour_statistics.cxx contour_statistics.h
  dice_statistics.cxx 
  )
SET (DICOM_INFO_SRC
    dicom_info.cxx)
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
SET (DLIB_TRAIN_SRC 
  dlib_train.cxx
  )
SET (DRR_SRC
  drr_main.c 
  drr_opts.c drr_opts.h
  )
SET (EXTRACT_CONTOUR_SRC
  extract_contour.cxx
  )
SET (FDK_SRC
  fdk_main.c 
  fdk_opts.c fdk_opts.h
  fdk_util.c fdk_util.h
)
IF (BROOK_FOUND)
  SET (FDK_SRC
    ${FDK_SRC}
    fdk_brook.cpp
    fdk_brook_kernel.br
    )
ENDIF ()
SET (FINDSCU_SRC
  findscu.cc
  )
SET (HND_TO_PFM_SRC
  hnd_to_pfm.c
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
ENDIF ()
WRAP_GGO (pcmd_segment_body_GGO_C pcmd_segment_body.ggo)
SET (PLASTIMATCH_SRC
  pcmd_add.cxx pcmd_add.h
  pcmd_adjust.cxx pcmd_adjust.h
  pcmd_autolabel.cxx pcmd_autolabel.h
  pcmd_compare.cxx pcmd_compare.h
  pcmd_compose.cxx pcmd_compose.h
  pcmd_crop.cxx pcmd_crop.h
  pcmd_diff.cxx pcmd_diff.h
  pcmd_dvh.cxx pcmd_dvh.h
  pcmd_xio_dvh.cxx pcmd_xio_dvh.h
  pcmd_header.cxx pcmd_header.h
  pcmd_mask.cxx pcmd_mask.h
  pcmd_resample.cxx pcmd_resample.h
  pcmd_segment.cxx pcmd_segment.h
  ${pcmd_segment_body_GGO_C}
  pcmd_stats.cxx pcmd_stats.h
  pcmd_thumbnail.cxx pcmd_thumbnail.h 
  pcmd_warp_dij.cxx 
  pcmd_warp.cxx pcmd_warp.h 
  pcmd_warp_pointset.cxx 
  plastimatch_main.cxx
  )
SET (PLASTIMATCH_SLICER_SRC
  plastimatch-slicer.cxx
  )
# NSh 1/3/2011
# Plugin for landmark warping based on standalone warper
# WRAP_GGO (landmark_warp_GGO_C landmark_warp.ggo)
SET (PLASTIMATCH_SLICER_LANDWARP_SRC
  plastimatch-slicer-landwarp.cxx
#  landmark_warp_args.cxx landmark_warp_args.h
#  ${landmark_warp_GGO_C}
  )
#SET (PLASTIMATCH_SLICER_TPS_SRC
# plastimatch-slicer-tps.cxx
#  )
SET (PLASTIMATCH_SLICER_DICOMRT_IMPORT_SRC
  plastimatch_slicer_dicomrt_import.cxx
  )
SET (PLASTIMATCH_SLICER_DICOMRT_EXPORT_SRC
  plastimatch_slicer_dicomrt_export.cxx
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
ENDIF ()
SET (PROTON_DOSE_SRC
  proton_dose_main.c 
  proton_dose_opts.c proton_dose_opts.h
  )
WRAP_GGO (landmark_warp_GGO_C landmark_warp.ggo)
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
SET (SYNTHETIC_MHA_DLIB_SRC
  synthetic_mha_main_dlib.cxx synthetic_mha_main.h
  )
SET (SYNTHETIC_MHA_TCLAP_SRC
  synthetic_mha_main_tclap.cxx synthetic_mha_main.h
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

##-----------------------------------------------------------------------------
##  SOURCE FILE PROPERTIES
##-----------------------------------------------------------------------------
IF (OPENMP_FOUND)
  SET_SOURCE_FILES_PROPERTIES (bspline.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (drr.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (fdk.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  SET_SOURCE_FILES_PROPERTIES (openmp_test.c
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
ENDIF ()

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
    ENDIF ()
    SET_SOURCE_FILES_PROPERTIES (
      bspline.c PROPERTIES COMPILE_FLAGS "${OLD_FLAGS} -msse2")
ENDIF ()

##-----------------------------------------------------------------------------
##  CONFIGURE INCLUDE FILES
##-----------------------------------------------------------------------------
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/plm_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/plm_api_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_api_config.h)

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/plm_fortran.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_fortran.h)

##-----------------------------------------------------------------------------
##  INCLUDE DIRECTORIES
##-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_BINARY_DIR})
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
ENDIF ()

IF (CUDA_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${CUDA_INCLUDE_DIRS})
  ##INCLUDE_DIRECTORIES (AFTER ${CUDA_CUT_INCLUDE_DIR})
ENDIF ()

IF (DCMTK_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${DCMTK_INCLUDE_DIR})
ENDIF ()

INCLUDE_DIRECTORIES (AFTER ${DLIB_INCLUDE_DIR})

IF (FFTW_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${FFTW_INCLUDE_DIR})
  LINK_DIRECTORIES (${FFTW_DIR})
ENDIF ()

IF (NOT HAVE_GETOPT_LONG)
  INCLUDE_DIRECTORIES (AFTER "${CMAKE_SOURCE_DIR}/libs/getopt")
ENDIF ()

IF (OPENCL_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${OPENCL_INCLUDE_DIR})
ENDIF ()

IF (MATLAB_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${MATLAB_INCLUDE_DIRS})
ENDIF ()

IF (PANTHEIOS_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${STLSOFT_INCLUDE_DIR})
  INCLUDE_DIRECTORIES (AFTER ${PANTHEIOS_INCLUDE_DIR})
ENDIF ()

IF (QT4_FOUND AND QT_QTSQL_FOUND)
  INCLUDE_DIRECTORIES (AFTER ${QT_QTSQL_INCLUDE_DIR})
ENDIF ()

IF (SLICER_FOUND)
  #MESSAGE (STATUS "SLICER INCLUDE DIRS: ${Slicer_Libs_INCLUDE_DIRS}")
  #MESSAGE (STATUS "SLICER INCLUDE DIRS: ${Slicer_Base_INCLUDE_DIRS}")
  INCLUDE_DIRECTORIES (AFTER
    ${SLICER_INCLUDE_DIRS}
    #${Slicer_Libs_INCLUDE_DIRS}
    #${Slicer_Base_INCLUDE_DIRS}
    )
ENDIF ()

IF (wxWidgets_FOUND)
  INCLUDE_DIRECTORIES (BEFORE ${wxWidgets_INCLUDE_DIRS})
ENDIF ()

##-----------------------------------------------------------------------------
##  Subdirectories
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (fatm/src)
ADD_SUBDIRECTORY (CUDA_tex_test)

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
  SET (GPUIT_LIB_LDFLAGS "${GPUIT_LIB_LDFLAGS} -ldl")
  SET (GPUIT_EXE_LDFLAGS "${GPUIT_EXE_LDFLAGS} -ldl")
  SET (PLMCUDA_EXE_LDFLAGS "${PLMCUDA_EXE_LDFLAGS} -ldl")
ENDIF ()
MESSAGE (STATUS "GPUIT_LIB_LDFLAGS: ${GPUIT_LIB_LDFLAGS}")
MESSAGE (STATUS "GPUIT_EXE_LDFLAGS: ${GPUIT_EXE_LDFLAGS}")
MESSAGE (STATUS "PLMCUDA_EXE_LDFLAGS: ${PLMCUDA_EXE_LDFLAGS}")

##-----------------------------------------------------------------------------
##  Library targets
##-----------------------------------------------------------------------------
PLM_ADD_LIBRARY (
  plmsys
  "${PLMSYS_LIBRARY_SRC}" 
  "${PLMSYS_LIBRARY_DEPENDENCIES}"
  "")

IF (F2C_LIBRARY AND UNIX)
  PLM_ADD_LIBRARY (f2c_helper "${F2C_HELPER_LIBRARY_SRC}" "" "")
ENDIF ()

PLM_ADD_LIBRARY (
  gpuit 
  "${GPUIT_LIBRARY_SRC}" 
  "${GPUIT_LIBRARY_DEPENDENCIES}"
  "${GPUIT_LIB_LDFLAGS}")

# JAS 2010.11.23
# Because the plugins are loaded at runtime, CMake doesn't know
# that they are needed by everything linked against gpuit.  So, we tell it.
IF (PLM_USE_GPU_PLUGINS)
  IF (CUDA_FOUND)
    ADD_DEPENDENCIES (gpuit plmcuda)
  ENDIF ()
  IF (OPENCL_FOUND)
    ADD_DEPENDENCIES (gpuit plmopencl)
  ENDIF ()
ENDIF ()

IF (ITK_FOUND)
  ## Despite claims to the contrary, you are not allowed to have a library
  ## with the same name as the executable.  What happens is the executable 
  ## tries to create a temporary library with the same name.  [MSVC 2005]
  ## That is why the library is called plastimatch1 instead of plastimatch.
  SET (PLASTIMATCH_LIBRARY_DEPENDENCIES ${ITK_LIBRARIES} ${GPUIT_LIBRARIES})
  PLM_ADD_LIBRARY (
    plastimatch1 
    "${PLASTIMATCH_LIBRARY_SRC}"
    "${PLASTIMATCH_LIBRARY_DEPENDENCIES}"
    "")
ENDIF ()

IF (WIN32 AND SLICER_FOUND)
  # The slicer plugin on windows can't find its dll easily.  Copy these 
  # over to the lib directory.
  
  SET (INDIR
    "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
  #"${CMAKE_BINARY_DIR}/Release")
  SET (OUTDIR
    "${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}/${CMAKE_CFG_INTDIR}")
  #"${CMAKE_BINARY_DIR}/lib/Slicer3/Plugins/Release")
  ADD_CUSTOM_TARGET (plastimatch_slicer_copy ALL
    DEPENDS "${OUTDIR}/plastimatch1.dll" "${OUTDIR}/gpuit.dll" 
    )
  ADD_CUSTOM_COMMAND (
    OUTPUT "${OUTDIR}/plastimatch1.dll"
    COMMAND
    ${CMAKE_COMMAND} "-E" "copy"
    "${INDIR}/plastimatch1.dll"
    "${OUTDIR}/plastimatch1.dll"
    DEPENDS plastimatch1
    )
  ADD_CUSTOM_COMMAND (
    OUTPUT "${OUTDIR}/gpuit.dll"
    COMMAND
    ${CMAKE_COMMAND} "-E" "copy"
    "${INDIR}/gpuit.dll"
    "${OUTDIR}/gpuit.dll"
    DEPENDS gpuit
    )
ENDIF ()

##-----------------------------------------------------------------------------
##  Executable targets
##-----------------------------------------------------------------------------
PLM_ADD_EXECUTABLE (bragg_curve "${BRAGG_CURVE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (bspline "${BSPLINE_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (check_grad "${CHECK_GRAD_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (demons "${DEMONS_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (drr "${DRR_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (dlib_train "${DLIB_TRAIN_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (fdk "${FDK_SRC}" "${GPUIT_LIBRARIES}" 
  "${PLMCUDA_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (hnd_to_pfm "${HND_TO_PFM_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 1)
PLM_ADD_EXECUTABLE (mha_to_raw mha_to_raw.c "" "" 1)
PLM_ADD_EXECUTABLE (proton_dose "${PROTON_DOSE_SRC}" "${GPUIT_LIBRARIES}" 
  "${GPUIT_EXE_LDFLAGS}" 0)
PLM_ADD_EXECUTABLE (raw_to_mha raw_to_mha.c "" "" 1)

IF (ITK_FOUND)
  MESSAGE (STATUS "PLASTIMATCH_LIBS: ${PLASTIMATCH_LIBS}")
  PLM_ADD_EXECUTABLE (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (compute_mean_image "${COMPUTE_MEAN_IMAGE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)

  PLM_ADD_EXECUTABLE (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (dicom_info "${DICOM_INFO_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (dicom_series_test "${DICOM_SERIES_TEST_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (distance_map "${DISTANCE_MAP_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (extract_contour "${EXTRACT_CONTOUR_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (merge_vfs merge_vector_fields.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (mha_to_rtog_dose "${MHA_TO_RTOG_DOSE_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (plastimatch "${PLASTIMATCH_SRC}"
    "${PLASTIMATCH_LIBS}" "${PLMCUDA_EXE_LDFLAGS}" 1)
  IF (SLICER_FOUND)
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer 
      "${PLASTIMATCH_SLICER_SRC}" "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}")
    PLM_ADD_SLICER_EXECUTABLE (plastimatch_slicer_dicomrt_import
      "${PLASTIMATCH_SLICER_DICOMRT_IMPORT_SRC}" "${PLASTIMATCH_LIBS}" 
      "${GPUIT_EXE_LDFLAGS}")
    PLM_ADD_SLICER_EXECUTABLE (plastimatch_slicer_dicomrt_export
      "${PLASTIMATCH_SLICER_DICOMRT_EXPORT_SRC}" "${PLASTIMATCH_LIBS}" 
      "${GPUIT_EXE_LDFLAGS}")
    PLM_ADD_SLICER_EXECUTABLE (plastimatch-slicer-landwarp 
      "${PLASTIMATCH_SLICER_LANDWARP_SRC}" "${PLASTIMATCH_LIBS}" 
      "${GPUIT_EXE_LDFLAGS}")
  ENDIF ()
  IF (QT4_FOUND AND QT_QTSQL_FOUND)
    SET (PLASTIMATCH_QT_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    PLM_ADD_EXECUTABLE (plastimatch_qt "${PLASTIMATCH_QT_SRC}"
      "${PLASTIMATCH_QT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 0)
  ENDIF ()
  PLM_ADD_EXECUTABLE (rtog_to_mha "${RTOG_TO_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
  PLM_ADD_EXECUTABLE (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (synthetic_mha "${SYNTHETIC_MHA_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  IF (PLM_BUILD_DLIB_TEST)
    PLM_ADD_EXECUTABLE (synthetic_mha_dlib "${SYNTHETIC_MHA_DLIB_SRC}"
      "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  ENDIF ()
  IF (PLM_BUILD_TCLAP_TEST)
    PLM_ADD_EXECUTABLE (synthetic_mha_tclap "${SYNTHETIC_MHA_TCLAP_SRC}"
      "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  ENDIF ()
  PLM_ADD_EXECUTABLE (landmark_warp "${LANDMARK_WARP_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (union_mask union_mask.cxx
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (vf_invert "${VF_INVERT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (xf_to_xf "${XF_TO_XF_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 1)
  PLM_ADD_EXECUTABLE (xf_to_EMPIREFMT "${XF_TO_EMPIREFMT_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 0)
ENDIF ()

# executables that require cuda
IF (CUDA_FOUND)
  PLM_ADD_EXECUTABLE (cuda_probe 
    "${CUDA_PROBE_SRC}" "${CUDA_LIBRARIES}" "" 1)
ENDIF ()

# executables that require opencl
IF (OPENCL_FOUND)
  PLM_ADD_EXECUTABLE (opencl_probe 
    "${OPENCL_PROBE_SRC}"  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 1)
ENDIF ()

# dicom_uid requires dcmtk
IF (DCMTK_FOUND)
  PLM_ADD_EXECUTABLE (dicom_uid 
    "${DICOM_UID_SRC}" "${DCMTK_LIBRARIES}" "" 1)
  PLM_ADD_EXECUTABLE (findscu "${FINDSCU_SRC}" "${DCMTK_LIBRARIES}" "" 0)
  PLM_ADD_EXECUTABLE (movescu "${MOVESCU_SRC}" "${DCMTK_LIBRARIES}" "" 0)
ENDIF ()

# mondoshot requires WIN32, wx, dcmtk, sqlite3
IF (WIN32 AND NOT CYGWIN AND wxWidgets_FOUND AND DCMTK_FOUND)
  ADD_EXECUTABLE (mondoshot WIN32 ${MONDOSHOT_SRC})
  TARGET_LINK_LIBRARIES (mondoshot 
    ${wxWidgets_LIBRARIES} ${DCMTK_LIBRARIES} ${SQLITE_LIBRARIES})
  IF (NOT BUILD_AGAINST_SLICER3)
    INSTALL (TARGETS mondoshot DESTINATION bin)
  ENDIF ()
ENDIF ()

# matlab mex files
IF (MATLAB_FOUND AND PLM_BUILD_MEX_TEST)
  MEX_TARGET (mex_drr 
    "${MEX_DRR_SRC}" "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}")
ENDIF ()

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
ENDIF ()

##-----------------------------------------------------------------------------
##  Additional install files
##-----------------------------------------------------------------------------
IF (ITK_FOUND)
  IF (EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")
    INSTALL(FILES "${ITK_DIR}/bin/release/ITKCommon.dll" DESTINATION bin)
  ENDIF (EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")
ENDIF ()

IF (CUDA_FOUND)
  IF (UNIX)
    SET (CUDART_LINUX_VERSION "libcudart.so.3.0.14")
    SET (CUDART_LINUX32 "${CUDA_TOOLKIT_ROOT_DIR}/lib/${CUDART_LINUX_VERSION}")
    SET (CUDART_LINUX64 "${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CUDART_LINUX_VERSION}")

    IF (HAVE_32_BIT)
      IF (EXISTS "${CUDART_LINUX32}")
        SET (CUDART_FILE_SRC "${CUDART_LINUX32}")
      ENDIF ()
    ENDIF ()

    IF (HAVE_64_BIT)
      IF (EXISTS "${CUDART_LINUX64}")
        SET (CUDART_FILE_SRC "${CUDART_LINUX64}")
      ENDIF ()
    ENDIF ()

    # Override for packagers building 32-bit packages on 64-bit machine
    IF (PLM_PACKAGE_32BIT)
      IF (EXISTS "${CUDART_LINUX32}")
        SET (CUDART_FILE_SRC "${CUDART_LINUX32}")
      ENDIF ()
    ENDIF ()

    # Hard code to cuda 3.0 (runtime 3.0.14).  Note, we copy it first, because
    # otherwise CMake 2.6 will install a broken symbolic link
    SET (CUDART_FILE "${CMAKE_BINARY_DIR}/${CUDART_LINUX_VERSION}")
    IF (EXISTS "${CUDART_FILE_SRC}")
      EXECUTE_PROCESS (COMMAND
        ${CMAKE_COMMAND} "-E" "copy" 
	    "${CUDART_FILE_SRC}" "${CUDART_FILE}"
      )
    ENDIF ()
  ELSE ()
    SET (CUDART_WIN32 "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32_30_14.dll")
    SET (CUDART_WIN64 "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_30_14.dll")

    IF (HAVE_32_BIT)
      IF (EXISTS "${CUDART_WIN32}")
        SET (CUDART_FILE "${CUDART_WIN32}")
      ENDIF ()
    ENDIF ()

    IF (HAVE_64_BIT)
      IF (EXISTS "${CUDART_WIN64}")
        SET (CUDART_FILE "${CUDART_WIN64}")
      ENDIF ()
    ENDIF ()

    # Override for packagers building 32-bit packages on 64-bit machine
    IF (PLM_PACKAGE_32BIT)
      IF (EXISTS "${CUDART_WIN32}")
        SET (CUDART_FILE_SRC "${CUDART_WIN32}")
      ENDIF ()
    ENDIF ()
  ENDIF ()

  IF (EXISTS "${CUDART_FILE}")
    IF (UNIX)
      INSTALL (FILES "${CUDART_FILE}" DESTINATION lib)
    ELSE ()
      INSTALL (FILES "${CUDART_FILE}" DESTINATION bin)
    ENDIF ()

    IF (BUILD_AGAINST_SLICER3)
      INSTALL (FILES "${CUDART_FILE}" DESTINATION lib/Slicer3/Plugins)
    ENDIF ()
  ENDIF ()
ENDIF ()

IF (FFTW_FOUND)
  IF (EXISTS "${FFTW_DIR}/libfftw3-3.dll")
    INSTALL (FILES "${FFTW_DIR}/libfftw3-3.dll" DESTINATION bin)
  ENDIF ()
ENDIF ()

## JAS 2011.01.24
## I have moved the documentation DESTINATION from
## /usr/doc to /usr/share/doc/plastimatch/ for Linux/UNIX
IF (NOT BUILD_AGAINST_SLICER3)
  IF (UNIX)
    # FHS compliant path for Linux/UNIX
    SET(DOC_DESTINATION "share/doc/plastimatch")
  ELSE ()
    # Just throw TXTs into a doc folder for Windows
    SET(DOC_DESTINATION "doc")
  ENDIF ()

  INSTALL (FILES 
    COPYRIGHT.TXT
    INSTALL.TXT
    LICENSE.TXT
    README.TXT
    README.BROOK.TXT
    REGISTRATION.TXT
    DESTINATION ${DOC_DESTINATION}
  )
ENDIF ()

##-----------------------------------------------------------------------------
##  MAKE UNINSTALL
##-----------------------------------------------------------------------------
IF (UNIX)
CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
ENDIF()

##-----------------------------------------------------------------------------
##  Simple test files
##-----------------------------------------------------------------------------
# Test executable -- cuda
IF (CUDA_FOUND AND PLM_BUILD_CUDA_TEST)
  CUDA_COMPILE (CUDA_TEST_WRAPPERS cuda_test.cu)
  SET (CUDA_TEST_SRC ${CUDA_TEST_WRAPPERS} cuda_test.cu)
  MESSAGE (STATUS "CUDA_TEST_SRC: ${CUDA_TEST_SRC}")
  PLM_ADD_EXECUTABLE (cuda_test 
    "${CUDA_TEST_SRC}" "${CUDA_LIBRARIES}" "" 0)
  SET_TARGET_PROPERTIES (cuda_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF ()

# Test executable -- dlib
IF (PLM_BUILD_DLIB_TEST)
  SET (DLIB_TEST_SRC dlib_test.cxx)
  SET (DLIB_TEST_LIBS "")
  PLM_ADD_EXECUTABLE (dlib_test 
    "${DLIB_TEST_SRC}" "${DLIB_TEST_LIBS}" "" 0)
ENDIF ()

# Test executable -- fann
IF (FANN_FOUND AND PLM_BUILD_FANN_TEST)
  SET (FANN_TEST_SRC fann_test.c)
  SET (FANN_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${FANN_LIBRARIES})
  PLM_ADD_EXECUTABLE (fann_test 
    "${FANN_TEST_SRC}" "${FANN_TEST_LIBS}" "${OPENMP_LDFLAGS}" 0)
ENDIF ()

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
  ENDIF ()
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
  ENDIF ()
ENDIF ()

# Test executable -- nlopt
IF (PLM_BUILD_NLOPT_TEST AND NLOPT_FOUND)
  SET (NLOPT_TEST_SRC nlopt_test.c)
  SET (NLOPT_TEST_LIBS 
    ${GPUIT_LIBRARIES} ${NLOPT_LIBRARIES})
  PLM_ADD_EXECUTABLE (nlopt_test "${NLOPT_TEST_SRC}" "${NLOPT_TEST_LIBS}" 
    "${OPENMP_LDFLAGS}" 0)
  SET_TARGET_PROPERTIES (nlopt_test PROPERTIES LINKER_LANGUAGE CXX)
ENDIF ()

# Test executable -- opencl
IF (PLM_BUILD_OPENCL_TEST AND OPENCL_FOUND)
  SET (OPENCL_TEST_SRC opencl_test.cxx opencl_test.h opencl_test.cl
    "${CMAKE_BINARY_DIR}/opencl_test.cl"
    )
  SET (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${PLASTIMATCH_LIBS})
  #SET (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${GPUIT_LIBRARIES})
  PLM_ADD_EXECUTABLE (opencl_test "${OPENCL_TEST_SRC}" 
    "${OPENCL_TEST_LIBS}" "" 0)
  #SET_TARGET_PROPERTIES (opencl_test PROPERTIES LINKER_LANGUAGE CXX)

  ## I don't yet know how to bundle the .cl file within the executable.
  ## Therefore, copy the .cl into binary directory.
  ADD_CUSTOM_COMMAND (
    OUTPUT "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_SOURCE_DIR}/opencl_test.cl" 
    "${CMAKE_BINARY_DIR}/opencl_test.cl" 
    DEPENDS "${CMAKE_SOURCE_DIR}/opencl_test.cl")
ENDIF ()

# Test executable -- openmp
IF (PLM_BUILD_OPENMP_TEST AND OPENMP_FOUND)
  SET (OPENMP_TEST_SRC openmp_test.c)
  PLM_ADD_EXECUTABLE (openmp_test "${OPENMP_TEST_SRC}" "${GPUIT_LIBRARIES}" 
    "${OPENMP_LDFLAGS}" 0)
ENDIF ()

# Test executable -- qt
IF (PLM_BUILD_QT_TEST AND QT4_FOUND)
  SET (QT_TEST_SRC qt_test.cxx)
  PLM_ADD_EXECUTABLE (qt_test "${QT_TEST_SRC}" "${QT_LIBRARIES}" "" 0)
ENDIF ()

# Test executable -- tclap
IF (PLM_BUILD_TCLAP_TEST)
  INCLUDE_DIRECTORIES (BEFORE "${CMAKE_SOURCE_DIR}/libs/tclap-1.2.0/include")
  SET (TCLAP_TEST_SRC tclap_test.cxx)
  PLM_ADD_EXECUTABLE (tclap_test "${TCLAP_TEST_SRC}" "" "" 0)
ENDIF ()

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
ELSE ()
  SET (PLM_PLASTIMATCH_PATH ${CMAKE_CURRENT_BINARY_DIR})
  SET (PLM_PLASTIMATCH_TESTING_PATH ${CMAKE_CURRENT_BINARY_DIR}/Testing)
  SET (PLM_FFTW_PATH "${FFTW_DIR}")
  SET (PLM_ITK_LIBRARY_PATH_HACK "")
  SET (PLM_PLASTIMATCH_PATH_HACK "")
  SET (PLM_FFTW_PATH_HACK "")
ENDIF ()

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
ENDMACRO ()

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
ENDIF ()

##-----------------------------------------------------------------------------
##  CPACK stuff
##  Note: "make package_source" to create tarball
##
##  JAS 2011.01.21
##  Extended the CPack stuff here for DEBs.
##
##  To generate...
##   * zip    : cpack -G ZIP <source tree>
##   * deb    : cpack -G DEB <source tree>
##   * tar.bz2: cpack -G TBZ2 <source tree>
##
##    ...from within your build directory
##-----------------------------------------------------------------------------

## Default Generators
SET (CPACK_GENERATOR "ZIP")
SET (CPACK_SOURCE_GENERATOR "TBZ2")

## Deb specific stuff
SET (CPACK_DEBIAN_PACKAGE_NAME "plastimatch")
IF (PLM_PACKAGE_32BIT)
  SET (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
ELSE ()
  IF (HAVE_32_BIT)
    SET (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
  ELSE ()
    SET (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  ENDIF ()
ENDIF ()
#SET (CPACK_DEBIAN_PACKAGE_DEPENDS "put depends here")
SET (CPACK_DEBIAN_PACKAGE_SECTION "science")
SET (CPACK_DEBIAN_PACKAGE_HOMEPAGE "www.plastimatch.org")
SET (CPACK_DEBIAN_PACKAGE_DESCRIPTION 
  "plastimatch\n Plastimatch is an open source software for deformable image
    registration. It is designed for high-performance volumetric registration
    of medical images, such as X-ray computed tomography, magnetic resonance
    imaging, and positron emission tomography."
)

## General Stuff
SET (CPACK_PACKAGE_CONTACT "gcsharp@partners.org")
SET (CPACK_PACKAGE_VERSION_MAJOR "1")
SET (CPACK_PACKAGE_VERSION_MINOR "4-beta")
SET (CPACK_PACKAGE_VERSION_PATCH ${PLASTIMATCH_SVN_VERSION})
SET (CPACK_SOURCE_IGNORE_FILES
  "/.svn"
  "~$"
)

INCLUDE (CPack)
