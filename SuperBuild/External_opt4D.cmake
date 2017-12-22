##-----------------------------------------------------------------------------
##  Download opt4D from internet and compile
##-----------------------------------------------------------------------------
set (proj opt4D)

if(NOT DEFINED git_protocol)
  set(git_protocol "git")
endif()

set (opt4D_GIT_REPOSITORY "${git_protocol}://gitlab.com/gregsharp/opt4D.git")
set (opt4D_GIT_TAG "2e5a9edcdc472c427ccfc2c567499a7405f02e47")

ExternalProject_Add (${proj}
  GIT_REPOSITORY "${opt4D_GIT_REPOSITORY}"
  GIT_TAG ${opt4D_GIT_TAG}
  SOURCE_DIR ${proj}
  BINARY_DIR ${proj}-build
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  #    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
  -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
  -DBUILD_EXAMPLES:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=ON
  -DBUILD_TESTING:BOOL=OFF
  -DModule_ITKReview:BOOL=ON
  INSTALL_COMMAND ""
  )

set (opt4D_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
