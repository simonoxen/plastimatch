project (bstrlib)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif ()

set (BSTRLIB_LIBRARY_SRC
#  bsafe.c
  bstraux.c
  bstrlib.c
  bstrwrap.cpp
  )
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
plm_add_static_library (bstrlib "${BSTRLIB_LIBRARY_SRC}" "" "" "")
