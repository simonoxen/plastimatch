PROJECT (inih)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC")
    set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fPIC")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
endif ()

SET (LIBINIH_SRC
    ini.c
    INIReader.cpp
    )

plm_add_static_library (inih "${LIBINIH_SRC}" "" "" "")
