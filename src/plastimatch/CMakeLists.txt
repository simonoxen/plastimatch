##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch)

##-----------------------------------------------------------------------------
##  Local options
##-----------------------------------------------------------------------------

# Plastimatch software configuration options
option (PLM_CONFIG_ALT_DCOS "Use alternative direction cosines rules" OFF)
option (PLM_CONFIG_KEYHOLIZE "Enable RT structure keyholization" OFF)
option (PLM_CONFIG_PREFER_DCMTK 
  "Use DCMTK instead of GDCM for DICOM operations" OFF)
option (PLM_USE_SS_IMAGE_VEC
  "Save structure sets as ITK UCHAR VEC images (in development)" OFF)
option (PLM_CONFIG_CLANG_COMPLETE "Generate .clang_complete for hipster Vim-ers" OFF)
option (PLM_CONFIG_DISABLE_VISCOUS "Disable experimental viscous fluid registration algorithm" ON)

##-----------------------------------------------------------------------------
##  Include directories
##-----------------------------------------------------------------------------
include_directories (BEFORE ${CMAKE_BINARY_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
# JAS 2012.04.25 !!!!!!!!!!! TEMPORARY !!!!!!!!!!!!!!!!!!!!!!
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/base)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cli)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/dose)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/opencl)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/qt)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/reconstruct)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/register)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/script)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/segment)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/standalone)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/sys)
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/util)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/itk-3.20.0)
include_directories (AFTER ${CMAKE_SOURCE_DIR}/libs/nSIFT)
include_directories (AFTER ${BSTRLIB_INCLUDE_DIR})
include_directories (AFTER ${DLIB_INCLUDE_DIR})
include_directories (AFTER ${LUA_INCLUDE_DIR})
include_directories (AFTER ${LIBLBFGS_INCLUDE_DIR})
include_directories (AFTER ${MSINTTYPES_INCLUDE_DIR})
include_directories (AFTER ${RANSAC_INCLUDE_DIRS})

if (CUDA_FOUND)
  include_directories (AFTER ${CUDA_INCLUDE_DIRS})
endif ()
if (DCMTK_FOUND)
  include_directories (AFTER ${DCMTK_INCLUDE_DIR})
endif ()
if (FFTW_FOUND)
  include_directories (BEFORE ${FFTW_INCLUDE_DIR})
  link_directories (${FFTW_DIR})
endif ()
if (NOT HAVE_GETOPT_LONG)
  include_directories (AFTER "${CMAKE_SOURCE_DIR}/libs/getopt")
endif ()
if (MATLAB_FOUND)
  include_directories (AFTER ${MATLAB_INCLUDE_DIRS})
endif ()
if (NLOPT_FOUND)
  include_directories (AFTER ${NLOPT_INCLUDE_DIR})
endif ()
if (OPENCL_FOUND)
  include_directories (BEFORE ${OPENCL_INCLUDE_DIRS})
endif ()
if (PANTHEIOS_FOUND)
  include_directories (AFTER ${STLSOFT_INCLUDE_DIR})
  include_directories (AFTER ${PANTHEIOS_INCLUDE_DIR})
endif ()
if (QT4_FOUND)
  if (QT_QTGUI_FOUND)
    include_directories (AFTER ${QT_QTGUI_INCLUDE_DIR})
  endif ()
  if (QT_QTSQL_FOUND)
    include_directories (AFTER ${QT_QTSQL_INCLUDE_DIR})
  endif ()
endif ()
if (SQLITE_FOUND)
  include_directories (AFTER ${SQLITE_INCLUDE_DIR})
endif ()

##-----------------------------------------------------------------------------
##  CONFIGURE INCLUDE FILES
##-----------------------------------------------------------------------------
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/sys/plm_config.h.in
  ${PLM_BUILD_ROOT}/plm_config.h)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/sys/plm_fortran.h.in
  ${PLM_BUILD_ROOT}/plm_fortran.h)

##-----------------------------------------------------------------------------
## Option to generate .clang_complete for hip Vim users using clang_complete
##-----------------------------------------------------------------------------
if (PLM_CONFIG_CLANG_COMPLETE)
    get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
    set (file_clang_complete "${PLM_SRC_ROOT}/src/plastimatch/.clang_complete")
    foreach (arg ${inc_dirs})
        set (inc_args "${inc_args} -I${arg}\n")
    endforeach ()
    file (WRITE "${file_clang_complete}" "${inc_args}")
    message (STATUS "Generated ${file_clang_complete}")
endif ()

##-----------------------------------------------------------------------------
##  Do we build a full plastimatch?
##-----------------------------------------------------------------------------
if (ITK_FOUND)
  if (GDCM_VERSION_2)
    message (WARNING "Plastimatch is built with GDCM 2.X -- DICOM-RT functions are disabled; if possible, please use GDCM 1.X instead.")
  endif ()
  set (FULL_PLASTIMATCH_BUILD 1)
endif ()

##-----------------------------------------------------------------------------
##  Do we have prerequisites for viscous code?
##-----------------------------------------------------------------------------
set (PLM_BUILD_VISCOUS FALSE)
if (NOT PLM_CONFIG_DISABLE_VISCOUS
    AND CUDA_FOUND AND CUDA_SDK_ROOT_DIR AND THRUST_FOUND)
  set (PLM_BUILD_VISCOUS TRUE)
endif ()


##-----------------------------------------------------------------------------
##  OLD BUILD SYSTEM REMNANTS
##-----------------------------------------------------------------------------
# JAS 2012.05.15 -- needs to be merged then removed for new build system
#if (FULL_PLASTIMATCH_BUILD)
#  # Ugh.  ITK is stupid.  They quietly dropped support for cygwin. 
#  # However, you can still get 3.20.0 to work.  You need to run 
#  # "make ITKFEM" after running "make" to workaround ITK's broken CMake 
#  # script:
#  #  http://old.nabble.com/compile-error-using-cygwin-td30879187.html
#  # See also:
#  #  http://public.kitware.com/Bug/view.php?id=11659
#  if (CYGWIN)
#    set (PLASTIMATCH1_LIBRARY_DEPENDENCIES 
#      ${PLASTIMATCH1_LIBRARY_DEPENDENCIES} gdi32)
#  endif ()
#endif ()

##-----------------------------------------------------------------------------
##  Special targets for slicer
##-----------------------------------------------------------------------------
if (SLICER_FOUND)
  # The slicer plugin on windows can't find its dll easily.  Copy these 
  # over to the lib directory.  Also, copy over the plastimatch 
  # executable 
  if (WIN32)
    set (INDIR "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
  else ()
    set (INDIR "${CMAKE_BINARY_DIR}")
  endif ()
  if (SLICER_IS_SLICER3)
    set (PLUGIN_DIR "${CMAKE_BINARY_DIR}/${Slicer3_INSTALL_PLUGINS_BIN_DIR}")
  else ()
    set (PLUGIN_DIR "${CMAKE_BINARY_DIR}/${Slicer_CLIMODULES_BIN_DIR}")
    set (QTPLUGIN_DIR "${CMAKE_BINARY_DIR}/${Slicer_QTLOADABLEMODULES_BIN_DIR}")
  endif ()
  set (OUTDIR "${PLUGIN_DIR}/${CMAKE_CFG_INTDIR}")
  set (QTOUTDIR "${QTPLUGIN_DIR}/${CMAKE_CFG_INTDIR}")

  # Copy plastimatch executable
  # GCS 2012-06-25: Don't do this any more.  It causes Slicer Debug 
  # build to barf.  Need to find another place to copy it...
  #  plm_add_target_copy (plastimatch_slicer_copy_plastimatch_exe
  #    "${INDIR}/plastimatch${CMAKE_EXECUTABLE_SUFFIX}"
  #    "${OUTDIR}/plastimatch${CMAKE_EXECUTABLE_SUFFIX}"
  #    plastimatch
  #    )

  # Copy dlls
  if (WIN32)
    if (CUDA_FOUND)
      plm_slicer_copy_dll (plastimatch_slicer_copy_plmcuda
	"${INDIR}/plmcuda.dll"
	"${OUTDIR}/plmcuda.dll"
	"${QTOUTDIR}/plmcuda.dll"
	plmcuda
	)
    endif ()
    if (OPENCL_FOUND)
      plm_slicer_copy_dll (plastimatch_slicer_copy_plmopencl
	"${INDIR}/plmopencl.dll"
	"${OUTDIR}/plmopencl.dll"
	"${QTOUTDIR}/plmopencl.dll"
	plmopencl
	)
    endif ()
  endif ()
endif ()

##-----------------------------------------------------------------------------
##  PLASTIMATCH MODULES
##-----------------------------------------------------------------------------
if (PLM_CONFIG_LIBRARY_BUILD)
else ()
  set (PLMLIB_CONFIG_ENABLE_CLI true)
  set (PLMLIB_CONFIG_ENABLE_CUDA true)
  set (PLMLIB_CONFIG_ENABLE_DOSE true)
  set (PLMLIB_CONFIG_ENABLE_OPENCL true)
  set (PLMLIB_CONFIG_ENABLE_QT true)
  set (PLMLIB_CONFIG_ENABLE_RECONSTRUCT true)
  set (PLMLIB_CONFIG_ENABLE_REGISTER true)
  set (PLMLIB_CONFIG_ENABLE_SCRIPT true)
  set (PLMLIB_CONFIG_ENABLE_SEGMENT true)
  set (PLMLIB_CONFIG_ENABLE_STANDALONE true)
  set (PLMLIB_CONFIG_ENABLE_TEST true)
endif ()

# Core
add_subdirectory(base)
add_subdirectory(sys)
add_subdirectory(util)

# Optional
if (PLMLIB_CONFIG_ENABLE_CLI)
  add_subdirectory(cli)
endif ()
if (CUDA_FOUND AND PLMLIB_CONFIG_ENABLE_CUDA)
  add_subdirectory(cuda)
endif ()
if (PLMLIB_CONFIG_ENABLE_DOSE)
  add_subdirectory(dose)
endif ()
if (OPENCL_FOUND AND PLMLIB_CONFIG_ENABLE_OPENCL)
  add_subdirectory(opencl)
endif ()
if (PLMLIB_CONFIG_ENABLE_QT)
  add_subdirectory(qt)
endif ()
if (PLMLIB_CONFIG_ENABLE_RECONSTRUCT)
  add_subdirectory(reconstruct)
endif ()
if (PLMLIB_CONFIG_ENABLE_REGISTER)
  add_subdirectory(register)
endif ()
if (PLMLIB_CONFIG_ENABLE_SCRIPT)
  add_subdirectory(script)
endif ()
if (PLMLIB_CONFIG_ENABLE_SEGMENT)
  add_subdirectory(segment)
endif ()
if (PLMLIB_CONFIG_ENABLE_STANDALONE)
  add_subdirectory(standalone)
endif ()
if (PLMLIB_CONFIG_ENABLE_TEST)
  add_subdirectory(test)
endif ()

##-----------------------------------------------------------------------------
##  DOXYGEN
##-----------------------------------------------------------------------------
if (DOXYGEN_FOUND)
  file (MAKE_DIRECTORY "${PLM_BUILD_ROOT}/doc")
  configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
    @ONLY)
  add_custom_target (doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY "${PLM_BUILD_ROOT}/doc"
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif ()
