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
  plm_add_static_library (specfun "${SPECFUN_LIBRARY_C_SRC}" 
    "${PLM_FORTRAN_LIBRARIES}" "" "")
else ()
  plm_add_static_library (specfun "${SPECFUN_LIBRARY_FORTRAN_SRC}" 
    "${PLM_FORTRAN_LIBRARIES}" "" "")
endif ()
