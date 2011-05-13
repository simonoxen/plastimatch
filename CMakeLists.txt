##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (plastimatch)

cmake_minimum_required (VERSION 2.6.0)

# Tell CMake to be quiet
# Note: it is ok to use CMake FindZLIB for 2.8.4.  Therefore setting 
# policy CMP0017 to NEW is safe.  But we do want the OLD policy for 
# older CMake versions.
cmake_policy (SET CMP0003 NEW)
if (NOT ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} VERSION_LESS 2.8.4)
  cmake_policy (SET CMP0017 NEW)
endif ()

# Default to release
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
endif ()

##-----------------------------------------------------------------------------
##  CMake include files
##-----------------------------------------------------------------------------
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include (cmake/PlmMacros.cmake)
include (CTest)
include (CheckFunctionExists)
include (CheckIncludeFiles)
include (CheckTypeSize)
include (TestBigEndian)

# http://www.cmake.org/pipermail/cmake/2008-December/025886.html
include (InstallRequiredSystemLibraries)

# http://www.cmake.org/Bug/view.php?id=9220
include (cmake/language_support.cmake)

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

# Plastimatch software configuration options
OPTION (PLM_USE_SS_IMAGE_VEC
  "Save structure sets as ITK UCHAR VECimages (in development)" OFF)
OPTION (PLM_CONFIG_DISABLE_CUDA "Set to ON to build without CUDA" OFF)
OPTION (PLM_CONFIG_DISABLE_OPENCL "Set to ON to build without OpenCL" OFF)
OPTION (PLM_SLICER_HARDCODED_FILENAME 
  "Use hardcoded filename instead of tmpfile()" OFF)
SET (PLM_CONFIG_VERSION_STRING "" CACHE STRING "Version string")

# Optional components
OPTION (PLM_BUILD_ISE "Build ISE fluoro program" OFF)
OPTION (PLM_BUILD_RTK "Build RTK" OFF)
OPTION (PLM_BUILD_SLICER_LOADABLE "Build slicer loadable module" OFF)

# Compile and link options
OPTION (BUILD_SHARED_LIBS 
  "Build plastimatch as shared library" OFF)
OPTION (PLM_PREFER_F2C 
  "Prefer using f2c over native fortran compiler" OFF)
OPTION (PLM_PREFER_SYSTEM_F2C 
  "Prefer the system f2c over the included f2c" ON)

# Installer Options
OPTION (PLM_INSTALL_RPATH 
  "Add full RPATH to install" ON)
OPTION (PLM_INSTALL_LIBRARIES
  "Include static libraries in install" OFF)
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
OPTION (PLM_TEST_DICOMRT_CERR
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
option (PLM_TEST_DICOMRT_IRREGULAR
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
##  Special hack for slicer extension on apple
##-----------------------------------------------------------------------------
IF (APPLE AND BUILD_AGAINST_SLICER3)
  SET (PLM_CONFIG_DISABLE_CUDA ON)
  SET (PLM_CONFIG_DISABLE_OPENCL ON)
ENDIF ()

##-----------------------------------------------------------------------------
##  Detect 32-bit or 64-bit machine
##-----------------------------------------------------------------------------
CHECK_TYPE_SIZE ("unsigned int"    CMAKE_SIZEOF_UINT)
CHECK_TYPE_SIZE ("unsigned long"   CMAKE_SIZEOF_ULONG)
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
##  Detect endian
##-----------------------------------------------------------------------------
TEST_BIG_ENDIAN (CMAKE_WORDS_BIGENDIAN)

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
## JAS 10.28.2010
## nvcc automatically passed the flag -malign-double to gcc when compiling .cu
## files.  This can be a problem when structs are shared between 
## .c & .cu files on 32-bit machines.  So, we pass -malign-double to gcc 
## for .c files as well so that everybody is using the same alignment.  
## -malign-double is automatically passed to gcc for 64-bit architectures.
## 
## GCS 2010-10-30
## -malign-double should only be enabled for 32-bit machines.  gcc 4.1.2 
## gives an error if it is used on x86_64.
##-----------------------------------------------------------------------------
IF (CMAKE_COMPILER_IS_GNUCC AND HAVE_32_BIT)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -malign-double")
    SET (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -malign-double")
ENDIF ()

##-----------------------------------------------------------------------------
##  Extra paths
##-----------------------------------------------------------------------------
SET (PLM_TESTING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Testing")
SET (PLM_TESTING_DATA_DIR "${CMAKE_SOURCE_DIR}/Testing/Data")
SET (PLM_TESTING_DOWNLOAD_DATA_DIR "${CMAKE_BINARY_DIR}/TestData")
SET (PLM_TESTING_BUILD_DIR "${CMAKE_BINARY_DIR}/Testing")

##-----------------------------------------------------------------------------
##  Search for include files
##-----------------------------------------------------------------------------
CHECK_INCLUDE_FILES ("stdint.h" HAVE_STDINT_H)
CHECK_INCLUDE_FILES ("sys/stat.h" HAVE_SYS_STAT_H)

##-----------------------------------------------------------------------------
##  Set the math library -- need this variable to configure libf2c
##-----------------------------------------------------------------------------
IF (UNIX)
  SET (MATH_LIB -lm)
ELSE ()
  SET (MATH_LIB)
ENDIF ()

##-----------------------------------------------------------------------------
##  Search for languages and language features
##-----------------------------------------------------------------------------
# Always check for fortran, whether we use it or not
IF (NOT MINGW)
  # For some reason, CMake tries to invoke devenv instead of gfortran
  # for mingw.
  INCLUDE (cmake/PLM_FindFortran.cmake)

  # The FortranCInterface is not included on CMake 2.6.2
  # INCLUDE (FortranCInterface)
ENDIF ()

# Choose between f2c or fortran
SET (PLM_USE_F2C FALSE)
IF (PLM_PREFER_F2C OR NOT CMAKE_Fortran_COMPILER_WORKS)
  SET (PLM_USE_F2C TRUE)
ENDIF ()

IF (PLM_USE_F2C)
  # If the user wants the system f2c, check for it
  IF (PLM_PREFER_SYSTEM_F2C)
    FIND_PACKAGE (F2C)
  ENDIF ()

  # If the user doesn't want system f2c, or didn't find it, build the 
  # included f2c
  IF (NOT PLM_PREFER_SYSTEM_F2C OR NOT F2C_LIBRARY)
    SET (PLM_USE_INCLUDED_F2C 1)
    ADD_SUBDIRECTORY (libs/libf2c)
    SET (F2C_LIBRARY f2c1)
    SET (F2C_INCLUDE_DIR libs/libf2c)
  ENDIF ()
ENDIF ()

# Some systems such as pcBSD don't have libdl
FIND_LIBRARY (LIBDL_FOUND dl)

##-----------------------------------------------------------------------------
##  Search for libraries
##-----------------------------------------------------------------------------
FIND_PACKAGE (Brook)
FIND_PACKAGE (DCMTK)
#FIND_PACKAGE (Dlib)
FIND_PACKAGE (Etags)
FIND_PACKAGE (Fann)
FIND_PACKAGE (FFTW)
FIND_PACKAGE (Liblbfgs)
FIND_PACKAGE (NLopt)
IF (PLM_BUILD_MEX_TEST)
  FIND_PACKAGE (Matlab)
ENDIF ()
FIND_PACKAGE (Octave)
IF (NOT PLM_CONFIG_DISABLE_OPENCL)
  FIND_PACKAGE (OpenCL)
ENDIF ()
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
##  Use local getopt if not found
##-----------------------------------------------------------------------------
check_function_exists (getopt_long HAVE_GETOPT_LONG)
if (NOT HAVE_GETOPT_LONG)
  set (GETOPT_LONG_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/getopt_test")
  set (GETOPT_LONG_TEST_C "${GETOPT_LONG_TEST_DIR}/getopt_test.c")
  file (MAKE_DIRECTORY "${GETOPT_LONG_TEST_DIR}")
  file (WRITE "${GETOPT_LONG_TEST_C}"
    "#include <getopt.h>\nmain(){getopt_long(0,0,0,0,0);}\n")
  try_compile (HAVE_GETOPT_LONG "${GETOPT_LONG_TEST_DIR}"
    "${GETOPT_LONG_TEST_C}")
endif ()
if (NOT HAVE_GETOPT_LONG)
  add_subdirectory (libs/getopt)
endif ()

##-----------------------------------------------------------------------------
##  Use local libsbfgs if not found
##-----------------------------------------------------------------------------
IF (NOT LIBLBFGS_FOUND)
  ADD_SUBDIRECTORY (libs/liblbfgs-1.9)
  SET (LIBLBFGS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/liblbfgs-1.9/include")
  SET (LIBLBFGS_LIBRARIES lbfgs)
  LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/liblbfgs-1.9)
  SET (LIBLBFGS_FOUND TRUE)
ENDIF ()

##-----------------------------------------------------------------------------
##  Use local sqlite if not found
##-----------------------------------------------------------------------------
IF (NOT SQLITE_FOUND)
  ADD_SUBDIRECTORY (libs/sqlite-3.6.21)
  SET (SQLITE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/sqlite-3.6.21")
  SET (SQLITE_LIBRARIES sqlite3)
ENDIF ()

##-----------------------------------------------------------------------------
##  Only use local bstrlib
##-----------------------------------------------------------------------------
ADD_SUBDIRECTORY (libs/bstrlib-05122010)
SET (BSTRLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/bstrlib-05122010")
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/libs/bstrlib-05122010)

##-----------------------------------------------------------------------------
##  Only use local dlib
##-----------------------------------------------------------------------------
SET (DLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/dlib-17.34")

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
##  Only use local msinttypes
##-----------------------------------------------------------------------------
SET (MSINTTYPES_INCLUDE_DIR 
  "${CMAKE_SOURCE_DIR}/libs" 
  )

##-----------------------------------------------------------------------------
##  Special CUDA processing
##-----------------------------------------------------------------------------
IF (NOT PLM_CONFIG_DISABLE_CUDA)
  FIND_PACKAGE (CUDA_wrap)
  SET (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")
  IF (CUDA_FOUND)
    CUDA_INCLUDE_DIRECTORIES (${CMAKE_CURRENT_SOURCE_DIR})
  ENDIF ()
ENDIF ()

##-----------------------------------------------------------------------------
##  Slicer
##-----------------------------------------------------------------------------
find_package (Slicer QUIET)
if (SLICER_FOUND)

  if (Slicer_USE_FILE)
    # Include USE file.  Strangely, the use file appends the 
    # Slicer directories to our include path.  This causes nvcc.exe 
    # to barf on windows, so we can't do this.

    # Equally strangely, the use file sets our C/CXX flags. 
    # These cause nvcc + gcc-4.3 to barf because the flags are for gcc-4.5
    set (OLD_CFLAGS ${CMAKE_C_FLAGS})
    set (OLD_CXXFLAGS ${CMAKE_CXX_FLAGS})
    get_directory_property (OLD_INCLUDE_DIR INCLUDE_DIRECTORIES)
    set_directory_properties (PROPERTIES INCLUDE_DIRECTORIES "")
    include ("${Slicer_USE_FILE}")
    get_directory_property (SLICER_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    set_directory_properties (PROPERTIES INCLUDE_DIRECTORIES
      "${OLD_INCLUDE_DIR}")
    set (CMAKE_C_FLAGS "${OLD_CFLAGS}")
    set (CMAKE_CXX_FLAGS "${OLD_CXXFLAGS}")
  endif ()

  # Set reasonable default install prefix and output paths
  # (after setting Slicer3_DIR, delete CMAKE_INSTALL_PREFIX and re-configure)
  if (SLICER_IS_SLICER3)
    slicer3_set_default_install_prefix_for_external_projects ()
    #slicer3_enable_plugins_testing()
  endif ()
else ()
  message (STATUS "Looking for Slicer - not found")
endif ()

##-----------------------------------------------------------------------------
##  For shared libraries, we enable dynamic loading of cuda, opencl
##  Note: we can't use BUILD_SHARED_LIBS directly, because it conflicts 
##  with the ITK header files
##  Note: we have to do this after checking for Slicer, because Slicer 
##  will change BUILD_SHARED_LIBS to ON
##-----------------------------------------------------------------------------
set (PLM_USE_GPU_PLUGINS OFF)
if (BUILD_SHARED_LIBS)
  set (PLM_USE_GPU_PLUGINS ON)
endif ()

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
##  Get build number from subversion 
##-----------------------------------------------------------------------------
SET (PLASTIMATCH_SVN_VERSION "EXPERIMENTAL")
IF (EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  IF (Subversion_FOUND)
    SET (PLASTIMATCH_HAVE_BUILD_NUMBER TRUE)
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
  SET (PLASTIMATCH_HAVE_BUILD_NUMBER FALSE)
ENDIF ()

##-----------------------------------------------------------------------------
##  ITK
##-----------------------------------------------------------------------------
find_package (ITK)
if (ITK_FOUND)
  include (${ITK_USE_FILE})
  set (ITK_VERSION 
    "${ITK_VERSION_MAJOR}.${ITK_VERSION_MINOR}.${ITK_VERSION_PATCH}")
  if (${ITK_VERSION} VERSION_LESS "3.16.0")
    message (FATAL_ERROR 
      "Fatal Error. ITK must be version 3.16.0 or higher")
  endif ()
  if (${ITK_VERSION_MAJOR} VERSION_EQUAL "4")
    message (STATUS "ITK is version 4.  You are brave!")
  endif ()
  if (${ITK_VERSION_MAJOR} VERSION_EQUAL "3" AND NOT ITK_USE_REVIEW)
    MESSAGE (FATAL_ERROR 
      "Fatal Error. ITK must be compiled with ITK_USE_REVIEW set to ON")
  endif ()
if (ITK_USE_SYSTEM_GDCM)
  message (FATAL_ERROR 
    "Fatal Error. ITK must be compiled with ITK_USE_SYSTEM_GDCM set to OFF")
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  Only use local ransac
##-----------------------------------------------------------------------------
IF (ITK_FOUND)
  ADD_SUBDIRECTORY (libs/ransac)
  SET (RANSAC_INCLUDE_DIRS 
    "${CMAKE_SOURCE_DIR}/libs/ransac" 
    "${CMAKE_SOURCE_DIR}/libs/ransac/Common")
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
##  CONFIGURE INCLUDE FILES
##-----------------------------------------------------------------------------
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/src/plastimatch/plm_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/src/plastimatch/plm_api_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_api_config.h)

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/src/plastimatch/plm_fortran.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_fortran.h)

##-----------------------------------------------------------------------------
##  Subdirectories
##-----------------------------------------------------------------------------
add_subdirectory (src)

##-----------------------------------------------------------------------------
##  Additional install files
##-----------------------------------------------------------------------------
IF (ITK_FOUND)
  IF (EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")
    INSTALL(FILES "${ITK_DIR}/bin/release/ITKCommon.dll" DESTINATION bin)
  ENDIF ()
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

# JAS 2011.01.24
# I have moved the documentation DESTINATION from
# /usr/doc to /usr/share/doc/plastimatch/ for Linux/UNIX
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
    DESTINATION ${DOC_DESTINATION}
  )
ENDIF ()

##-----------------------------------------------------------------------------
##  MAKE UNINSTALL
##-----------------------------------------------------------------------------
if (UNIX)
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

add_custom_target (uninstall COMMAND 
  ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()

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
  # CMake doesn't allow "=" to be passed in a -D parameter.  So we substitute 
  # with replacement string, which will get substituted back within the 
  # cmake script itself
  STRING (REPLACE "=" "&equal&" TMP_PARMS "${PARMS}")
  ADD_TEST (${TESTNAME} ${CMAKE_COMMAND} 
    -DTESTNAME=${TESTNAME}
    -DWORKING_DIR=${WORKING_DIR}
    ${PLM_ITK_LIBRARY_PATH_HACK}
    ${PLM_PLASTIMATCH_PATH_HACK}
    ${PLM_FFTW_PATH_HACK}
    -DPLM_TEST_COMMAND=${PLM_TEST_COMMAND}
    -DPLM_TESTING_SOURCE_DIR=${PLM_TESTING_SOURCE_DIR}
    -DPLM_TESTING_BUILD_DIR=${PLM_TESTING_BUILD_DIR}
    "-DPARMS=${TMP_PARMS}"
    -P ${CMAKE_SOURCE_DIR}/cmake/RUN_CTEST.cmake
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

# Default Generators
SET (CPACK_GENERATOR "ZIP")
SET (CPACK_SOURCE_GENERATOR "TBZ2")

# Deb specific stuff
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

# General Stuff
SET (CPACK_PACKAGE_CONTACT "gcsharp@partners.org")
SET (CPACK_PACKAGE_VERSION_MAJOR "1")
SET (CPACK_PACKAGE_VERSION_MINOR "5.0-beta")
SET (CPACK_PACKAGE_VERSION_PATCH ${PLASTIMATCH_SVN_VERSION})
SET (CPACK_SOURCE_IGNORE_FILES
  "/.svn"
  "~$"
)

INCLUDE (CPack)
