project (specfun)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
  set (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -fPIC")
  set (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -fPIC")
endif ()


# For plm_fortran.h
include_directories (BEFORE ${CMAKE_BINARY_DIR})

# We only need this one
set (SPECFUN_LIBRARY_FORTRAN_SRC
  mpbdv.for
  )
set (SPECFUN_LIBRARY_C_SRC
  mpbdv.c
  )

if (PLM_USE_F2C)
  #add_library (specfun STATIC ${SPECFUN_LIBRARY_C_SRC})
  plm_add_library (specfun "${SPECFUN_LIBRARY_C_SRC}" 
    "${PLM_FORTRAN}" "" "")
else ()
  #add_library (specfun STATIC ${SPECFUN_LIBRARY_FORTRAN_SRC})
  plm_add_library (specfun "${SPECFUN_LIBRARY_FORTRAN_SRC}" 
    "${PLM_FORTRAN}" "" "")
endif ()

## JAS 2011.01.24
## I have disabled this INSTALL since it would be inappropriate
## for our CPack packages to install libspecfun.a
##
#if (PLM_OLD_INSTALL_TARGETS)
#  install (TARGETS specfun DESTINATION lib)
#endif ()

