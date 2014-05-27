project (dbf1)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
  set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif ()

include_directories ("${DBF_INCLUDE_DIR}")

set (DBFCORE_LIB_SRC
  an_string.c
  an_string.h
  codepages.c
  codepages.h
  congraph.c
  congraph.h
  csv.c
  csv.h
  odbf.c
  odbf.h
  sql.c
  sql.h
  statistic.c
  statistic.h
)

set (DBF1_SRC
  dbf_main.c
  dbf_main.h
)

plm_add_static_library (dbfcore "${DBFCORE_LIB_SRC}" "" "" "")

plm_add_executable (dbf1 "${DBF1_SRC}" 
  "${DBF_LIBRARIES}" "" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
