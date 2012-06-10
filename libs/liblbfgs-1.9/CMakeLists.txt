project (liblbfgs)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif ()

set (LIBLBFGS_LIBRARY_SRC
  lib/lbfgs.c
  lib/arithmetic_ansi.h
  lib/arithmetic_sse_float.h
  lib/arithmetic_sse_double.h
  )
include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#add_library (lbfgs STATIC ${LIBLBFGS_LIBRARY_SRC})
plm_add_library (lbfgs "${LIBLBFGS_LIBRARY_SRC}" "" "" "")

## JAS 2011.01.24
## I have disabled this INSTALL since it would be inappropriate
## for our CPack packages to install libsqlite3.a
##
# IF (PLM_OLD_INSTALL_TARGETS)
#   INSTALL (TARGETS lbfgs DESTINATION lib)
# ENDIF (PLM_OLD_INSTALL_TARGETS)
