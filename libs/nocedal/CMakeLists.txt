project (nocedal)

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
set (NOCEDAL_LIBRARY_FORTRAN_SRC
  lbfgsb_2_1.f
  )
set (NOCEDAL_LIBRARY_C_SRC
  lbfgsb_2_1.c
  )

if (PLM_USE_F2C)
  add_library (nocedal STATIC ${NOCEDAL_LIBRARY_C_SRC})
else ()
  add_library (nocedal STATIC ${NOCEDAL_LIBRARY_FORTRAN_SRC})
endif ()

## JAS 2011.01.24
## I have disabled this INSTALL since it would be inappropriate
## for our CPack packages to install libnocedal.a
##
if (PLM_OLD_INSTALL_TARGETS)
  install (TARGETS nocedal DESTINATION lib)
endif (PLM_OLD_INSTALL_TARGETS)
