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

if (PLM_PREFER_NO_FORTRAN_NO_F2C OR PLM_USE_F2C)
  plm_add_static_library (nocedal "${NOCEDAL_LIBRARY_C_SRC}" 
    "${PLM_FORTRAN_LIBRARIES}" "" "")
else ()
  plm_add_static_library (nocedal "${NOCEDAL_LIBRARY_FORTRAN_SRC}" 
    "${PLM_FORTRAN_LIBRARIES}" "" "")
endif ()
