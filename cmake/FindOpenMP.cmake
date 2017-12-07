######################################################
##  OpenMP
######################################################
include (CheckFunctionExists)
include (CheckCXXSourceCompiles)
message (STATUS "Check for compiler OpenMP support...")
set (OPENMP_FLAGS)
set (OPENMP_LIBRARIES)
set (OPENMP_FOUND FALSE)

# sample openmp source code to test
set(OpenMP_C_TEST_SOURCE
"
#include <omp.h>
int main() {
#ifdef _OPENMP
  return 0;
#else
  breaks_on_purpose
#endif
}
")

# Key: CFLAGS##LDFLAGS#LIBRARIES
# Neither CFLAGS nor LDFLAGS can be empty.  Use NONE instead.
set(
  OPENMP_FLAGS_AND_LIBRARIES
  # gcc
  "-fopenmp##-fopenmp#"
  "-fopenmp##-fopenmp#gomp"
  "-fopenmp##-fopenmp#gomp pthread"
  # MSVC
  "/openmp##NONE#"
  # clang (??)
  "-fopenmp=libomp##NONE#gomp"
  # icc
  "-openmp##-openmp#"
  "-openmp -parallel##-openmp -parallel#"
  # SGI & PGI
  "-mp##-mp#"
  # Sun
  "-xopenmp##-xopenmp#"
  # Tru64
  "-omp##-omp#"
  # AIX
  "-qsmp=omp##-qsmp=omp#"
)

# Massive hack to workaround CMake limitations
list (LENGTH OPENMP_FLAGS_AND_LIBRARIES NUM_FLAGS)
math (EXPR NUM_FLAGS "${NUM_FLAGS} - 1")
foreach (I RANGE 0 ${NUM_FLAGS})
  if (NOT OPENMP_FOUND)
    list (GET OPENMP_FLAGS_AND_LIBRARIES ${I} TMP)
    string (REGEX MATCH "([^#]*)" OPENMP_FLAGS ${TMP})
    string (REGEX REPLACE "[^#]*##" "" TMP ${TMP})
    string (REGEX MATCH "([^#]*)" OPENMP_LDFLAGS ${TMP})
    string (REGEX REPLACE "[^#]*#" "" OPENMP_LIBRARIES ${TMP})
#    message (STATUS "OPENMP_FLAGS=${OPENMP_FLAGS}")
#    message (STATUS "OPENMP_LDFLAGS = ${OPENMP_LDFLAGS}")
#    message (STATUS "OPENMP_LIBRARIES = ${OPENMP_LIBRARIES}")
#    message (STATUS "-------")

    if (OPENMP_LDFLAGS MATCHES "NONE")
      set (OPENMP_LDFLAGS "")
    endif ()
    if (OPENMP_LIBRARIES MATCHES " ")
      string (REPLACE " " ";" OPENMP_LIBRARIES ${OPENMP_LIBRARIES})
    endif ()

    # 2017-12-07. If you overwrite CMAKE_REQUIRED_FLAGS, FindCUDA 
    # does not work correctly.  It gives the wrong set of libraries.
    # Therefore, we must save and restore the original values.
    push_vars ("CMAKE_REQUIRED_QUIET" "CMAKE_REQUIRED_FLAGS" 
        "CMAKE_REQUIRED_LIBRARIES")
    set (CMAKE_REQUIRED_QUIET TRUE)
    set (CMAKE_REQUIRED_FLAGS ${OPENMP_FLAGS})
    set (CMAKE_REQUIRED_LIBRARIES ${OPENMP_LIBRARIES})

    # CMake caches results from test compilations.  We need to unset the 
    # cache value, or else cached test results gets used after first 
    # iteration
    unset (OPENMP_COMPILES CACHE)

    check_cxx_source_compiles ("${OpenMP_C_TEST_SOURCE}" OPENMP_COMPILES)

    pop_vars ("CMAKE_REQUIRED_QUIET" "CMAKE_REQUIRED_FLAGS" 
        "CMAKE_REQUIRED_LIBRARIES")

    if (OPENMP_COMPILES)
      set (OPENMP_FOUND TRUE)
    endif ()
  endif ()
endforeach ()

if (OPENMP_FOUND)
  message (STATUS "OpenMP flags \"${OPENMP_FLAGS}\", OpenMP libraries \"${OPENMP_LIBRARIES}\"")
else ()
  message (STATUS "Given compiler does not support OpenMP.")
endif ()
