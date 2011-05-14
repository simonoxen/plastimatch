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
option (PLM_BUILD_CUDA_TEST "Build CUDA test program" OFF)
option (PLM_BUILD_DLIB_TEST "Build dlib test program" OFF)
option (PLM_BUILD_FANN_TEST "Build fann test program" OFF)
option (PLM_BUILD_MEX_TEST "Build mex test program" OFF)
option (PLM_BUILD_NLOPT_TEST "Build NLopt test program" OFF)
option (PLM_BUILD_OPENCL_TEST "Build OpenCL test program" OFF)
option (PLM_BUILD_OPENMP_TEST "Build OpenMP test program" OFF)
option (PLM_BUILD_QT_TEST "Build Qt test program" OFF)

# Plastimatch software configuration options
option (PLM_USE_SS_IMAGE_VEC
  "Save structure sets as ITK UCHAR VECimages (in development)" OFF)
option (PLM_CONFIG_DISABLE_CUDA "Set to ON to build without CUDA" OFF)
option (PLM_CONFIG_DISABLE_OPENCL "Set to ON to build without OpenCL" OFF)
option (PLM_SLICER_HARDCODED_FILENAME 
  "Use hardcoded filename instead of tmpfile()" OFF)
set (PLM_CONFIG_VERSION_STRING "" CACHE STRING "Version string")

# Optional components
option (PLM_BUILD_ISE "Build ISE fluoro program" OFF)
option (PLM_BUILD_RTK "Build RTK" OFF)
option (PLM_BUILD_SLICER_LOADABLE "Build slicer loadable module" OFF)

# Compile and link options
option (BUILD_SHARED_LIBS 
  "Build plastimatch as shared library" OFF)
option (PLM_PREFER_F2C 
  "Prefer using f2c over native fortran compiler" OFF)
option (PLM_PREFER_SYSTEM_F2C 
  "Prefer the system f2c over the included f2c" ON)

# Installer Options
option (PLM_INSTALL_RPATH 
  "Add full RPATH to install" ON)
option (PLM_INSTALL_LIBRARIES
  "Include static libraries in install" OFF)
option (PLM_PACKAGE_32BIT 
  "Set this when building 32-bit packages on a 64-bit machine" OFF)

# Regression testing
option (PLM_BUILD_TESTING 
  "Enable regression testing" ON)
option (PLM_REDUCED_TESTS 
  "Run fewer tests (used when debugging tests)" OFF)
option (PLM_TEST_BSPLINE_FLAVORS 
  "Regression tests for seldom used b-spline implementations" OFF)
option (PLM_TEST_DICOMRT_AW
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
option (PLM_TEST_DICOMRT_CERR
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
option (PLM_TEST_DICOMRT_IRREGULAR
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
option (PLM_TEST_DICOMRT_PINNACLE 
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
option (PLM_TEST_DICOMRT_XIO 
  "Regression test for dicom, dicomrt (downloads extra test data)" OFF)
option (PLM_TEST_FDK_VARIAN 
  "Regression tests for fdk with Varian data (downloads extra test data)" OFF)
option (PLM_TEST_XIO 
  "Regression tests for xio file exchange (downloads extra test data)" OFF)

##-----------------------------------------------------------------------------
##  Special hack for slicer extension on apple
##-----------------------------------------------------------------------------
if (APPLE AND BUILD_AGAINST_SLICER3)
  set (PLM_CONFIG_DISABLE_CUDA ON)
  set (PLM_CONFIG_DISABLE_OPENCL ON)
endif ()

##-----------------------------------------------------------------------------
##  Detect 32-bit or 64-bit machine
##-----------------------------------------------------------------------------
check_type_size ("unsigned int"    CMAKE_SIZEOF_UINT)
check_type_size ("unsigned long"   CMAKE_SIZEOF_ULONG)
if (NOT APPLE)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set (HAVE_32_BIT TRUE)
    set (HAVE_64_BIT FALSE)
    message (STATUS "Machine is 32-bit")
  else ()
    set (HAVE_32_BIT FALSE)
    set (HAVE_64_BIT TRUE)
    message (STATUS "Machine is 64-bit")
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  Detect endian
##-----------------------------------------------------------------------------
test_big_endian (CMAKE_WORDS_BIGENDIAN)

##-----------------------------------------------------------------------------
##  I can't remember why this is necessary.  But at one time I needed 
##  it for compiling on AMD linux 64-bit.
##-----------------------------------------------------------------------------
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif ()

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
if (CMAKE_COMPILER_IS_GNUCC AND HAVE_32_BIT)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -malign-double")
  set (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -malign-double")
endif ()

##-----------------------------------------------------------------------------
##  Extra paths
##-----------------------------------------------------------------------------
set (PLM_TESTING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/Testing")
set (PLM_TESTING_DATA_DIR "${CMAKE_SOURCE_DIR}/Testing/Data")
set (PLM_TESTING_DOWNLOAD_DATA_DIR "${CMAKE_BINARY_DIR}/TestData")
set (PLM_TESTING_BUILD_DIR "${CMAKE_BINARY_DIR}/Testing")

##-----------------------------------------------------------------------------
##  Search for include files
##-----------------------------------------------------------------------------
check_include_files ("stdint.h" HAVE_STDINT_H)
check_include_files ("sys/stat.h" HAVE_SYS_STAT_H)

##-----------------------------------------------------------------------------
##  Set the math library -- need this variable to configure libf2c
##-----------------------------------------------------------------------------
if (UNIX)
  set (MATH_LIB -lm)
else ()
  set (MATH_LIB)
endif ()

##-----------------------------------------------------------------------------
##  Search for languages and language features
##-----------------------------------------------------------------------------
# Always check for fortran, whether we use it or not
if (NOT MINGW)
  # For some reason, CMake tries to invoke devenv instead of gfortran
  # for mingw.
  include (cmake/PLM_FindFortran.cmake)

  # The FortranCInterface is not included on CMake 2.6.2
  # INCLUDE (FortranCInterface)
endif ()

# Choose between f2c or fortran
set (PLM_USE_F2C FALSE)
if (PLM_PREFER_F2C OR NOT CMAKE_Fortran_COMPILER_WORKS)
  set (PLM_USE_F2C TRUE)
endif ()

if (PLM_USE_F2C)
  # If the user wants the system f2c, check for it
  if (PLM_PREFER_SYSTEM_F2C)
    find_package (F2C)
  endif ()

  # If the user doesn't want system f2c, or didn't find it, build the 
  # included f2c
  if (NOT PLM_PREFER_SYSTEM_F2C OR NOT F2C_LIBRARY)
    set (PLM_USE_INCLUDED_F2C 1)
    add_subdirectory (libs/libf2c)
    set (F2C_LIBRARY f2c1)
    set (F2C_INCLUDE_DIR libs/libf2c)
  endif ()
endif ()

# Some systems such as pcBSD don't have libdl
find_library (LIBDL_FOUND dl)

##-----------------------------------------------------------------------------
##  Search for libraries
##-----------------------------------------------------------------------------
find_package (Brook)
find_package (DCMTK)
#FIND_PACKAGE (Dlib)
find_package (Etags)
find_package (Fann)
find_package (FFTW)
find_package (Liblbfgs)
find_package (NLopt)
if (PLM_BUILD_MEX_TEST)
  find_package (Matlab)
endif ()
find_package (Octave)
if (NOT PLM_CONFIG_DISABLE_OPENCL)
  find_package (OpenCL)
endif ()
find_package (OpenGL)
find_package (OpenMP)
find_package (Qt4 4.6.2 COMPONENTS QtCore QtGui OPTIONAL)
if (PLM_BUILD_RTK)
  find_package (RTK)
endif ()
find_package (SQLite)
find_package (SSE)           # SSE Extensions for CPU
find_package (Subversion)
find_package (wxWidgets)

##-----------------------------------------------------------------------------
##  QT requires including the use file
##-----------------------------------------------------------------------------
if (QT4_FOUND)
  message (STATUS "Looking for Qt4 - found")
  include (${QT_USE_FILE})
else ()
  message (STATUS "Looking for Qt4 - not found")
endif ()

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
if (NOT LIBLBFGS_FOUND)
  add_subdirectory (libs/liblbfgs-1.9)
  set (LIBLBFGS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/liblbfgs-1.9/include")
  set (LIBLBFGS_LIBRARIES lbfgs)
  link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/liblbfgs-1.9)
  set (LIBLBFGS_FOUND TRUE)
endif ()

##-----------------------------------------------------------------------------
##  Use local sqlite if not found
##-----------------------------------------------------------------------------
if (NOT SQLITE_FOUND)
  add_subdirectory (libs/sqlite-3.6.21)
  set (SQLITE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/sqlite-3.6.21")
  set (SQLITE_LIBRARIES sqlite3)
endif ()

##-----------------------------------------------------------------------------
##  Only use local bstrlib
##-----------------------------------------------------------------------------
add_subdirectory (libs/bstrlib-05122010)
set (BSTRLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/bstrlib-05122010")
link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/bstrlib-05122010)

##-----------------------------------------------------------------------------
##  Only use local dlib
##-----------------------------------------------------------------------------
set (DLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libs/dlib-17.34")

##-----------------------------------------------------------------------------
##  Only use local nocedal
##-----------------------------------------------------------------------------
add_subdirectory (libs/nocedal)
link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/nocedal)

##-----------------------------------------------------------------------------
##  Only use local specfun
##-----------------------------------------------------------------------------
add_subdirectory (libs/specfun)
link_directories (${CMAKE_CURRENT_BINARY_DIR}/libs/specfun)

##-----------------------------------------------------------------------------
##  Only use local msinttypes
##-----------------------------------------------------------------------------
set (MSINTTYPES_INCLUDE_DIR 
  "${CMAKE_SOURCE_DIR}/libs" 
  )

##-----------------------------------------------------------------------------
##  Special CUDA processing
##-----------------------------------------------------------------------------
if (NOT PLM_CONFIG_DISABLE_CUDA)
  find_package (CUDA_wrap)
  set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")
  if (CUDA_FOUND)
    cuda_include_directories (${CMAKE_CURRENT_SOURCE_DIR})
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  Slicer
##-----------------------------------------------------------------------------
find_package (Slicer QUIET)
if (SLICER_FOUND)

  # Strangely, the Slicer use file (both Slicer 3 and Slicer 4) 
  # appends the Slicer directories to our include path.  
  # This causes nvcc.exe to barf on windows, so we can't let it do this.
  # 
  # Equally strangely, in Slicer 4, the use file sets our C/CXX flags. 
  # These cause nvcc + gcc-4.3 to barf because the flags are for gcc-4.5
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
if (PLM_INSTALL_RPATH)
  # use, i.e. don't skip the full RPATH for the build tree
  set (CMAKE_SKIP_BUILD_RPATH  FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  set (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

  # the RPATH to be used when installing
  set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif ()

##-----------------------------------------------------------------------------
##  User feedback on which languages and libraries were found
##-----------------------------------------------------------------------------
if (wxWidgets_FOUND)
  message (STATUS "Looking for wxWidgets - found.")
else ()
  message (STATUS "Looking for wxWidgets - not found.")
endif ()
if (Matlab_FOUND)
  message (STATUS "Looking for Matlab - found.")
else ()
  message (STATUS "Looking for Matlab - not found.")
endif ()
if (OCTAVE_FOUND)
  message (STATUS "Looking for Octave - found.")
else ()
  message (STATUS "Looking for Octave - not found.")
endif ()

##-----------------------------------------------------------------------------
##  Get build number from subversion 
##-----------------------------------------------------------------------------
set (PLASTIMATCH_SVN_VERSION "EXPERIMENTAL")
if (EXISTS "${CMAKE_SOURCE_DIR}/.svn")
  if (Subversion_FOUND)
    set (PLASTIMATCH_HAVE_BUILD_NUMBER TRUE)
    #Subversion_WC_INFO(${CMAKE_SOURCE_DIR} "svn")
    execute_process (
      COMMAND 
      ${Subversion_SVNVERSION_EXECUTABLE} "--no-newline" ${CMAKE_SOURCE_DIR}
      RESULT_VARIABLE svn_result
      OUTPUT_VARIABLE svn_version
      ERROR_VARIABLE svn_error
      )
    if (${svn_result} EQUAL 0)
      message (STATUS "Looking for version in svn - found")
      set (PLASTIMATCH_SVN_VERSION ${svn_version})
    else ()
      message (STATUS "Looking for version in svn - can't connect")
    endif ()
  else ()
    message (STATUS "Looking for version in svn - svn not found")
  endif ()
else ()
  set (PLASTIMATCH_HAVE_BUILD_NUMBER FALSE)
endif ()

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
    message (FATAL_ERROR 
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
if (ITK_FOUND)
  add_subdirectory (libs/ransac)
  set (RANSAC_INCLUDE_DIRS 
    "${CMAKE_SOURCE_DIR}/libs/ransac" 
    "${CMAKE_SOURCE_DIR}/libs/ransac/Common")
endif ()

##-----------------------------------------------------------------------------
##  WXWINDOWS LIBRARY
##-----------------------------------------------------------------------------
if (wxWidgets_FOUND)
  if (wxWidgets_CXX_FLAGS)
    string (REGEX REPLACE "-D" "" wxWidgets_DEFINITIONS 
      "${wxWidgets_DEFINITIONS}")
    set_source_files_properties (mondoshot_main.cpp
      PROPERTIES COMPILE_FLAGS 
      ${wxWidgets_CXX_FLAGS}
      )
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  CONFIGURE INCLUDE FILES
##-----------------------------------------------------------------------------
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/src/plastimatch/plm_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_config.h)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/src/plastimatch/plm_api_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_api_config.h)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/src/plastimatch/plm_fortran.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/plm_fortran.h)

##-----------------------------------------------------------------------------
##  Subdirectories
##-----------------------------------------------------------------------------
add_subdirectory (src)

##-----------------------------------------------------------------------------
##  Additional install files
##-----------------------------------------------------------------------------
if (ITK_FOUND)
  if (EXISTS "${ITK_DIR}/bin/release/ITKCommon.dll")
    install (FILES "${ITK_DIR}/bin/release/ITKCommon.dll" DESTINATION bin)
  endif ()
endif ()

if (CUDA_FOUND)
  if (UNIX)
    set (CUDART_LINUX_VERSION "libcudart.so.3.0.14")
    set (CUDART_LINUX32 
      "${CUDA_TOOLKIT_ROOT_DIR}/lib/${CUDART_LINUX_VERSION}")
    set (CUDART_LINUX64 
      "${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CUDART_LINUX_VERSION}")

    if (HAVE_32_BIT AND EXISTS "${CUDART_LINUX32}")
      set (CUDART_FILE_SRC "${CUDART_LINUX32}")
    endif ()

    if (HAVE_64_BIT AND EXISTS "${CUDART_LINUX64}")
      set (CUDART_FILE_SRC "${CUDART_LINUX64}")
    endif ()

    # Override for packagers building 32-bit packages on 64-bit machine
    if (PLM_PACKAGE_32BIT AND EXISTS "${CUDART_LINUX32}")
      set (CUDART_FILE_SRC "${CUDART_LINUX32}")
    endif ()

    # Hard code to cuda 3.0 (runtime 3.0.14).  Note, we copy it first, because
    # otherwise CMake 2.6 will install a broken symbolic link
    set (CUDART_FILE "${CMAKE_BINARY_DIR}/${CUDART_LINUX_VERSION}")
    if (EXISTS "${CUDART_FILE_SRC}")
      execute_process (COMMAND
        ${CMAKE_COMMAND} "-E" "copy" 
	    "${CUDART_FILE_SRC}" "${CUDART_FILE}"
      )
    endif ()
  else ()
    set (CUDART_WIN32 "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32_30_14.dll")
    set (CUDART_WIN64 "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_30_14.dll")

    if (HAVE_32_BIT AND EXISTS "${CUDART_WIN32}")
      set (CUDART_FILE "${CUDART_WIN32}")
    endif ()

    if (HAVE_64_BIT AND EXISTS "${CUDART_WIN64}")
      set (CUDART_FILE "${CUDART_WIN64}")
    endif ()

    # Override for packagers building 32-bit packages on 64-bit machine
    if (PLM_PACKAGE_32BIT AND EXISTS "${CUDART_WIN32}")
      set (CUDART_FILE_SRC "${CUDART_WIN32}")
    endif ()
  endif ()

  if (EXISTS "${CUDART_FILE}")
    if (UNIX)
      install (FILES "${CUDART_FILE}" DESTINATION lib)
    else ()
      install (FILES "${CUDART_FILE}" DESTINATION bin)
    endif ()

    if (BUILD_AGAINST_SLICER3)
      install (FILES "${CUDART_FILE}" DESTINATION lib/Slicer3/Plugins)
    endif ()
  endif ()
endif ()

if (FFTW_FOUND)
  if (EXISTS "${FFTW_DIR}/libfftw3-3.dll")
    install (FILES "${FFTW_DIR}/libfftw3-3.dll" DESTINATION bin)
  endif ()
endif ()

# JAS 2011.01.24
# I have moved the documentation DESTINATION from
# /usr/doc to /usr/share/doc/plastimatch/ for Linux/UNIX
if (NOT BUILD_AGAINST_SLICER3)
  if (UNIX)
    # FHS compliant path for Linux/UNIX
    set (DOC_DESTINATION "share/doc/plastimatch")
  else ()
    # Just throw TXTs into a doc folder for Windows
    set (DOC_DESTINATION "doc")
  endif ()

  install (FILES 
    COPYRIGHT.TXT
    INSTALL.TXT
    LICENSE.TXT
    README.TXT
    DESTINATION ${DOC_DESTINATION}
  )
endif ()

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
