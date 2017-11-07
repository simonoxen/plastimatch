##-----------------------------------------------------------------------------
##  Download ITK from internet and compile
##-----------------------------------------------------------------------------
message (STATUS "Hello from External_ITK")

set (proj ITK)

set (itk_url https://downloads.sourceforge.net/project/itk/itk/4.12/InsightToolkit-4.12.2.tar.gz)
set (itk_md5sum 758206eeb458d11b7ba2d81d8a3ce212)

ExternalProject_Add (${proj}
  DOWNLOAD_DIR ${proj}-download
  URL ${itk_url}
  URL_MD5 ${itk_md5sum}
#  PATCH_COMMAND "${CMAKE_COMMAND}" 
#    -DPLM_SOURCE_DIR=${CMAKE_SOURCE_DIR}
#    -DPLM_TARGET_DIR=${CMAKE_BINARY_DIR}/${proj}
#    -P "${CMAKE_SOURCE_DIR}/cmake/ExternalITKPatch.cmake"
  SOURCE_DIR ${proj}
  BINARY_DIR ${proj}-build
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
#    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DBUILD_TESTING:BOOL=OFF
#    -DITK_USE_REVIEW:BOOL=ON
#    -DITK_USE_REVIEW_STATISTICS:BOOL=ON
#    -DITK_USE_OPTIMIZED_REGISTRATION_METHODS:BOOL=ON
  -DModule_ITKReview:BOOL=ON
  INSTALL_COMMAND ""
  )

set (ITK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
