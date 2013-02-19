project (nkidecompress)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif ()

set (NKIDECOMPRESS_SRC
  nkidecompress.cxx
  )

##add_library (nkidecompress STATIC ${NKIDECOMPRESS_SRC})
plm_add_static_library (nkidecompress "${NKIDECOMPRESS_SRC}" "" "" "")
