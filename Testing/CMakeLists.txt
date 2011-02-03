PROJECT(PLM_Testing)

## -------------------------------------------------------------------------
## Download data sets
## -------------------------------------------------------------------------
MACRO (PLM_DOWNLOAD_AND_EXTRACT DOWNLOAD_URL DOWNLOAD_FILENAME DOWNLOAD_DIR)
  IF (NOT EXISTS ${DOWNLOAD_FILENAME})
    MESSAGE (STATUS "Downloading test data")
    FILE (DOWNLOAD "${DOWNLOAD_URL}" "${DOWNLOAD_FILENAME}")
  ENDIF (NOT EXISTS ${DOWNLOAD_FILENAME})
  IF (NOT EXISTS ${DOWNLOAD_DIR})
    MESSAGE (STATUS "Untarring test data")
    EXECUTE_PROCESS (
      COMMAND 
      ${CMAKE_COMMAND} "-E" "tar" "xvzf" "${DOWNLOAD_FILENAME}"
      WORKING_DIRECTORY ${PLM_TESTING_BUILD_DIR}
      RESULT_VARIABLE untar_result
      OUTPUT_VARIABLE untar_output
      ERROR_VARIABLE untar_error)
    MESSAGE (STATUS "Untar command returned: ${untar_result}\n${untar_output}\n${untar_error}")
  ENDIF (NOT EXISTS ${DOWNLOAD_DIR})
ENDMACRO (PLM_DOWNLOAD_AND_EXTRACT)

IF (PLM_TEST_DICOMRT_XIO)
  PLM_DOWNLOAD_AND_EXTRACT (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/934/chest-phantom-dicomrt-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")
ENDIF (PLM_TEST_DICOMRT_XIO)

IF (PLM_TEST_DICOMRT_PINNACLE)
  PLM_DOWNLOAD_AND_EXTRACT (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1187/rando-dicomrt-pinnacle3-8.2g.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/rando-dicomrt-pinnacle3-8.2g.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/rando-dicomrt-pinnacle3-8.2g")
ENDIF (PLM_TEST_DICOMRT_PINNACLE)

IF (PLM_TEST_FDK_VARIAN)
  PLM_DOWNLOAD_AND_EXTRACT (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1127/varian-catphan.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/varian-catphan.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/varian-catphan")
ENDIF (PLM_TEST_FDK_VARIAN)

IF (PLM_TEST_XIO)
  PLM_DOWNLOAD_AND_EXTRACT (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/939/chest-phantom-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")
ENDIF (PLM_TEST_XIO)

## -------------------------------------------------------------------------
## Copy plugin library
##   GCS: N.b. I don't think this is necessary.  There is no need to 
##   copy gpuit or plastimatch1 libraries.  Instead we set the extra path 
##   as an argument to PLM_ADD_TEST.
## -------------------------------------------------------------------------
# JAS 2010.11.23
# Tests will fail if the test binaries cannot find the cuda plugin.
# So, we copy it into the testing directory.  This seems to work well.
IF (PLM_USE_GPU_PLUGINS)
    # Because MSVC places stuff in Release/ or Debug/, etc
    # CMAKE_CFG_INTDIR is "Release/" or "Debug/" for Windows or "/" for Linux
    # In the Linux case, the extra "/" doesn't hurt anything.
    SET (GPU_PLUGIN_BIN_DIR
        "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/")

    IF (CUDA_FOUND)
        # CMAKE_SHARED_LIBRARY_PREFIX is "" under Windows and "lib" under Linux
        # CMAKE_SHARE_LIBRARY_SUFFIX is ".dll" under Windows and ".so" under Linux
        SET (CUDA_PLUGIN_FILE
            "${CMAKE_SHARED_LIBRARY_PREFIX}plmcuda${CMAKE_SHARED_LIBRARY_SUFFIX}")

        # This tells CMake how to handle a target depending on
        # "Testing/[lib]plmcuda[.dll/.so]"
        ADD_CUSTOM_COMMAND (
            OUTPUT "${PLM_TESTING_BUILD_DIR}/${CUDA_PLUGIN_FILE}"
            COMMAND
            ${CMAKE_COMMAND} "-E" "copy"
            "${GPU_PLUGIN_BIN_DIR}/${CUDA_PLUGIN_FILE}"
            "${PLM_TESTING_BUILD_DIR}/${CUDA_PLUGIN_FILE}"
            DEPENDS plmcuda
        )

        # The Phony target
        # This is the Makefile equivalent of
        # Testing/plmcuda.dll : plmcuda.dll      (Under Windows)  *OR*
        # Testing/libplmcuda.so : libplmcuda.so  (Under Linux)
        #
        # The "rule" for handling this is the above custom command 
        ADD_CUSTOM_TARGET (plmcuda_tcpy
            DEPENDS "${PLM_TESTING_BUILD_DIR}/${CUDA_PLUGIN_FILE}"
        )

        # When using the plmcuda plugin, GPUIT depends on it for GPU acceleration.
        # So, we add it as a dependency here so that plmcuda plugin will be copied
        # to Testing whenever the user builds GPUIT (or any target that depends on
        # GPUIT, which is most targets).
        ADD_DEPENDENCIES (gpuit plmcuda_tcpy)
    ENDIF (CUDA_FOUND)

    IF (OPENCL_FOUND)
        SET (OPENCL_PLUGIN_FILE
            "${CMAKE_SHARED_LIBRARY_PREFIX}plmopencl${CMAKE_SHARED_LIBRARY_SUFFIX}")

        ADD_CUSTOM_COMMAND (
            OUTPUT "${PLM_TESTING_BUILD_DIR}/${OPENCL_PLUGIN_FILE}"
            COMMAND
            ${CMAKE_COMMAND} "-E" "copy"
            "${GPU_PLUGIN_BIN_DIR}/${OPENCL_PLUGIN_FILE}"
            "${PLM_TESTING_BUILD_DIR}/${OPENCL_PLUGIN_FILE}"
            DEPENDS plmopencl
        )

        ADD_CUSTOM_TARGET (plmopencl_tcpy
            DEPENDS "${PLM_TESTING_BUILD_DIR}/${OPENCL_PLUGIN_FILE}"
        )

        ADD_DEPENDENCIES (gpuit plmopencl_tcpy)
    ENDIF (OPENCL_FOUND)
ENDIF (PLM_USE_GPU_PLUGINS)

## -------------------------------------------------------------------------
## Test code targets
## -------------------------------------------------------------------------
ADD_EXECUTABLE (bspline_test_1 bspline_test_1.cxx)
IF (NOT ${GPUIT_EXE_LDFLAGS} STREQUAL "")
  SET_TARGET_PROPERTIES(bspline_test_1 
    PROPERTIES LINK_FLAGS ${GPUIT_EXE_LDFLAGS})
ENDIF (NOT ${GPUIT_EXE_LDFLAGS} STREQUAL "")
TARGET_LINK_LIBRARIES (bspline_test_1 ${GPUIT_LIBRARIES})

ADD_EXECUTABLE (plm_drr_api_a plm_drr_api_a.cxx)
IF (NOT ${GPUIT_EXE_LDFLAGS} STREQUAL "")
  SET_TARGET_PROPERTIES (plm_drr_api_a 
    PROPERTIES LINK_FLAGS ${GPUIT_EXE_LDFLAGS})
ENDIF (NOT ${GPUIT_EXE_LDFLAGS} STREQUAL "")
TARGET_LINK_LIBRARIES (plm_drr_api_a ${PLASTIMATCH_LIBS})

ADD_EXECUTABLE (plm_registration_api_a plm_registration_api_a.cxx)
IF (NOT ${GPUIT_EXE_LDFLAGS} STREQUAL "")
  SET_TARGET_PROPERTIES (plm_registration_api_a 
    PROPERTIES LINK_FLAGS ${GPUIT_EXE_LDFLAGS})
ENDIF (NOT ${GPUIT_EXE_LDFLAGS} STREQUAL "")
TARGET_LINK_LIBRARIES (plm_registration_api_a ${PLASTIMATCH_LIBS})


## -------------------------------------------------------------------------
## synth-test - create synthetic mha files
##   gauss-1.mha   Centered gauss
##   gauss-2.mha   Off-center gauss
##   gauss-3.mha   Inverted gauss
##   gauss-4.mha   Gauss with inverted pixel spacing
##   gauss-5.mha   Even more off-center gauss
##   gauss-6.mha   Off-center gauss, in off-center world coordinates
##   gauss-ushort-1.mha   Centered gauss, ushort
##   gauss-ushort-2.mha   Off-center gauss, ushort
##   rect-1.mha    Inverted rect, centered
##   rect-2.mha    Standard rect, off-center
##   rect-3.mha    Standard rect, centered
##   sphere-1.mha  Uchar sphere, bg=0, fg=255
## -------------------------------------------------------------------------
SET (SYNTH_MHA_SIZE 38)

PLM_ADD_TEST (
  "gauss-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-2.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-3"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-3.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;-1000;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-4"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-4.mha;--output-type;float;--pattern;gauss;--origin;+25 +25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--spacing;-1.31579 -1.31579 2;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-5"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;20 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-6"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-6.mha;--output-type;float;--pattern;gauss;--origin;-15 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-ushort-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-ushort-1.mha;--output-type;ushort;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;1000;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "gauss-ushort-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/gauss-ushort-2.mha;--output-type;ushort;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;1000;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "rect-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/rect-1.mha;--output-type;float;--pattern;rect;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;0;--foreground;-1000"
  )

PLM_ADD_TEST (
  "rect-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/rect-2.mha;--output-type;float;--pattern;rect;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--rect-size;-30 70 -50 50 -50 50"
  )

PLM_ADD_TEST (
  "rect-3"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/rect-3.mha;--output-type;float;--pattern;rect;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--rect-size;-50 50 -50 50 -50 50"
  )

PLM_ADD_TEST (
  "sphere-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/synthetic_mha
  "--output;${PLM_TESTING_BUILD_DIR}/sphere-1.mha;--output-type;uchar;--pattern;sphere;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;255;--sphere-center;0 0 0;--sphere-radius;10 10 10"
  )

## -------------------------------------------------------------------------
## bspline-bxf - read and write bxf format
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "bspline-bxf"
  ""
  ${PLM_PLASTIMATCH_TESTING_PATH}/bspline_test_1
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt"
  )
ADD_TEST (bspline-bxf-check ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt"
  "${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt")

## -------------------------------------------------------------------------
## bspline - bspline algorithm flavors
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "bspline-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;a;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-a-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-a.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=18945.5"
  "-DUPPER_THRESH=18945.7"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;b;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-b-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-b.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=18945.5"
  "-DUPPER_THRESH=18945.7"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;c;-x;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-c-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-c.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=1168.25"
  "-DUPPER_THRESH=1168.35"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-d"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;d;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-d-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-d.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=1168.25"
  "-DUPPER_THRESH=1168.35"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-e"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;e;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-e-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-e.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=1168.25"
  "-DUPPER_THRESH=1168.35"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-f"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;f;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-f-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-f.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=1168.25"
  "-DUPPER_THRESH=1168.35"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-g"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;g;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-g-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-g.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=1168.25"
  "-DUPPER_THRESH=1168.35"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-h"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;h;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (bspline-h-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-h.stdout.txt"
  "-DREGEX=^MSE\\\\[.*, 10] *([0-9.]*)"
  "-DLOWER_THRESH=1168.25"
  "-DUPPER_THRESH=1168.35"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-mi-c-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;c;-M;mi;-O;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-3.mha"
  )
ADD_TEST (bspline-mi-c-1-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1.stdout.txt"
  "-DREGEX=^MI\\\\[.*, 10] *([-0-9.]*)"
  "-DLOWER_THRESH=-0.800"
  "-DUPPER_THRESH=-0.796"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-mi-c-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;c;-M;mi;-O;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-2.mha;${PLM_TESTING_BUILD_DIR}/rect-1.mha;${PLM_TESTING_BUILD_DIR}/rect-2.mha"
  )
## Without DOUBLE_HIST defined, output is 0.031
## With DOUBLE_HIST defined, output is 0.032
ADD_TEST (bspline-mi-c-2-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-mi-c-2.stdout.txt"
  "-DREGEX=^MI\\\\[.*, 10] *([-0-9.]*)"
  "-DLOWER_THRESH=-0.035"
  "-DUPPER_THRESH=-0.031"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## demons
##   demons-a          gauss-1.mha, gauss-2.mha
##   demons-cuda-a     gauss-1.mha, gauss-2.mha
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "demons-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/demons
  "-m;50;-s;3;-f;5 5 5;-O;${PLM_TESTING_BUILD_DIR}/demons-a.mha;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (demons-a-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/demons-a.stdout.txt"
  "-DREGEX=^Mean: *([0-9.]*)"
  "-DLOWER_THRESH=2.1"
  "-DUPPER_THRESH=2.2"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "demons-cuda-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/demons
  "-A;cuda;-m;50;-s;3;-f;5 5 5;-O;${PLM_TESTING_BUILD_DIR}/demons-cuda-a.mha;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
ADD_TEST (demons-cuda-a-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/demons-cuda-a.stdout.txt"
  "-DREGEX=^Mean: *([0-9.]*)"
  "-DLOWER_THRESH=2.1"
  "-DUPPER_THRESH=2.2"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )


## -------------------------------------------------------------------------
## drr
##   drr-a       gauss-1.mha (normal image), exact
##   drr-b       gauss-1.mha (normal image), uniform
##   drr-c       gauss-4.mha (negative spacing), exact
##   drr-d       gauss-4.mha (negative spacing), uniform
##   drr-cuda    gauss-1.mha, uniform
##   drr-opencl  gauss-1.mha, uniform
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "drr-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-a/out_;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
PLM_ADD_TEST (
  "drr-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-a/out_0010.pfm"
  )
ADD_TEST (
  "drr-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-a-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=0.045"
  "-DUPPER_THRESH=0.055"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "drr-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-b/out_;-i;uniform;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
PLM_ADD_TEST (
  "drr-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-b/out_0010.pfm"
  )
ADD_TEST (
  "drr-b-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-b-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=0.045"
  "-DUPPER_THRESH=0.055"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "drr-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-c/out_;${PLM_TESTING_BUILD_DIR}/gauss-4.mha"
  )
PLM_ADD_TEST (
  "drr-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-c/out_0010.pfm"
  )
ADD_TEST (
  "drr-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-c-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=0.045"
  "-DUPPER_THRESH=0.055"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "drr-d"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-d/out_;-i;uniform;${PLM_TESTING_BUILD_DIR}/gauss-4.mha"
  )
PLM_ADD_TEST (
  "drr-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-d/out_0010.pfm"
  )
ADD_TEST (
  "drr-d-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-d-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=0.045"
  "-DUPPER_THRESH=0.055"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "drr-cuda"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-A;cuda;-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-cuda/out_;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
PLM_ADD_TEST (
  "drr-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/drr-cuda/out_0010.pfm"
  )
ADD_TEST (
  "drr-cuda-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-cuda-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=0.045"
  "-DUPPER_THRESH=0.055"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "drr-opencl"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-A;opencl;-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-opencl/out_;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
PLM_ADD_TEST (
  "drr-opencl-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/drr-opencl/out_0010.pfm"
  )
ADD_TEST (
  "drr-opencl-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-opencl-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=0.045"
  "-DUPPER_THRESH=0.055"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## fdk - all tests using fdk filtered backprojection
##   fdk-cpu-a     CPU reconstruction of synthetic data
##   fdk-cpu-b     CPU reconstruction of Varian data, no filtering
##   fdk-cpu-c     CPU reconstruction of Varian data, fftw filtering
##   fdk-brook     GPU reconstruction of synthetic data using BROOK
##   fdk-cuda-a    GPU reconstruction of synthetic data using CUDA
##    [fdk-cuda-b  GPU reconstruction of Varian data using CUDA
##   fdk-opencl-a  GPU reconstruction of synthetic data using OpenCL
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-cpu-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk-cpu-a.mha"
  )
PLM_ADD_TEST (
  "fdk-cpu-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;fdk-cpu-a.mha"
  )
ADD_TEST (
  "fdk-cpu-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-cpu-a-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=-987.5"
  "-DUPPER_THRESH=-986.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "fdk-cpu-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/varian-catphan;-f;none;-a;0 20 670;-O;${PLM_TESTING_BUILD_DIR}/fdk-cpu-b.mha"
  )
PLM_ADD_TEST (
  "fdk-cpu-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;fdk-cpu-b.mha"
  )
ADD_TEST (
  "fdk-cpu-b-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-cpu-b-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-821.5"
  "-DUPPER_THRESH=-820.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "fdk-cpu-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/varian-catphan;-f;ramp;-a;0 10 670;-O;${PLM_TESTING_BUILD_DIR}/fdk-cpu-c.mha"
  )
PLM_ADD_TEST (
  "fdk-cpu-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;fdk-cpu-c.mha"
  )
ADD_TEST (
  "fdk-cpu-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-cpu-c-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-990.0"
  "-DUPPER_THRESH=-989.0"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "fdk-brook"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;brook;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk_brook_output.mha"
  )
PLM_ADD_TEST (
  "fdk-brook-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;fdk_brook_output.mha"
  )
ADD_TEST (
  "fdk-brook-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-brook-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=-987.5"
  "-DUPPER_THRESH=-986.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "fdk-cuda-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;cuda;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk-cuda-a.mha"
  )
PLM_ADD_TEST (
  "fdk-cuda-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;fdk-cuda-a.mha"
  )
ADD_TEST (
  "fdk-cuda-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-cuda-a-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=-987.5"
  "-DUPPER_THRESH=-986.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "fdk-opencl-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;opencl;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk-opencl-a.mha"
  )
PLM_ADD_TEST (
  "fdk-opencl-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;fdk-opencl-a.mha"
  )
ADD_TEST (
  "fdk-opencl-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-opencl-a-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=-987.5"
  "-DUPPER_THRESH=-986.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## landmark_warp a: itk tps, slicer fcsv
## landmark_warp b: wendland rbf, slicer fcsv
## landmark_warp c: nsh gauss rbf, slicer fcsv
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "landmark-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/landmark_warp
  "-a;tps;-f;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;-m;${PLM_TESTING_DATA_DIR}/fiducials-rect-3.fcsv;-I;${PLM_TESTING_BUILD_DIR}/rect-3.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp-a-img.mha;-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-a-vf.mha"
  )
PLM_ADD_TEST (
  "landmark-warp-a-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;landmark-warp-a-img.mha"
  )
ADD_TEST (
  "landmark-warp-a-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/landmark-warp-a-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-860.5"
  "-DUPPER_THRESH=-860.0"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "landmark-warp-a-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;landmark-warp-a-vf.mha"
  )
ADD_TEST (
  "landmark-warp-a-check-2"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/landmark-warp-a-stats-2.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=-23.2"
  "-DUPPER_THRESH=-22.7"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "landmark-warp-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/landmark_warp
  "-a;wendland;-f;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;-m;${PLM_TESTING_DATA_DIR}/fiducials-rect-3.fcsv;-I;${PLM_TESTING_BUILD_DIR}/rect-3.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp-b-img.mha;-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-b-vf.mha"
  )
PLM_ADD_TEST (
  "landmark-warp-b-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;landmark-warp-b-img.mha"
  )
ADD_TEST (
  "landmark-warp-b-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/landmark-warp-b-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-992.0"
  "-DUPPER_THRESH=-991.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "landmark-warp-b-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;landmark-warp-b-vf.mha"
  )
ADD_TEST (
  "landmark-warp-b-check-2"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/landmark-warp-b-stats-2.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=-1.55"
  "-DUPPER_THRESH=-1.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "landmark-warp-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/landmark_warp
  "-a;gauss;-f;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;-m;${PLM_TESTING_DATA_DIR}/fiducials-rect-3.fcsv;-I;${PLM_TESTING_BUILD_DIR}/rect-3.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-img.mha;-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-vf.mha;-r;75"
  )
PLM_ADD_TEST (
  "landmark-warp-c-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;landmark-warp-c-img.mha"
  )
ADD_TEST (
  "landmark-warp-c-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/landmark-warp-c-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-992.0"
  "-DUPPER_THRESH=-991.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "landmark-warp-c-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;landmark-warp-c-vf.mha"
  )
ADD_TEST (
  "landmark-warp-c-check-2"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/landmark-warp-c-stats-2.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=-1.6"
  "-DUPPER_THRESH=-1.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch convert dicom (and dicom rt)
##   plm convert dicom a: ct only
##   plm convert dicom b: rtss only, with dicom-dir reference
##   plm convert dicom c: rtss only, with fixed image reference
##   plm convert dicom d: dose only, native sampling rate
##   plm convert dicom e: input xio dicom, output plastimatch dicom
##   plm convert dicom f: input plastimatch dicom, output plastimatch dicom
##   plm convert dicom g: input rtss, no fixed image, no ct reference
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-convert-dicom-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output;plm-convert-dicom-a.mha"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-convert-dicom-a.mha"
  )
ADD_TEST (
  "plm-convert-dicom-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-667.5"
  "-DUPPER_THRESH=-666.6"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-convert-dicom-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;plm-convert-dicom-b.mha;--dicom-dir;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-convert-dicom-b.mha"
  )
ADD_TEST (
  "plm-convert-dicom-b-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-b-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=1.05"
  "-DUPPER_THRESH=1.06"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-convert-dicom-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;plm-convert-dicom-c.mha;--fixed;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a.mha;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c.cxt"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-convert-dicom-c.mha"
  )
ADD_TEST (
  "plm-convert-dicom-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=1.05"
  "-DUPPER_THRESH=1.06"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-convert-dicom-d"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/DOSE.20080627A.TRAINING4FLD.dcm;--output-dose-img;plm-convert-dicom-d.mha"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-convert-dicom-d.mha"
  )
ADD_TEST (
  "plm-convert-dicom-d-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-d-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=42.1"
  "-DUPPER_THRESH=42.2"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-convert-dicom-e" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dicom;--output-dose-img;plm-convert-dicom-e-dose-img.mha"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-f" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dicom;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-dicom;--output-dose-img;plm-convert-dicom-f-dose-img.mha"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-f-stats" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compare;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dose-img.mha;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-dose-img.mha"
  )
ADD_TEST (
  "plm-convert-dicom-f-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-stats.stdout.txt"
  "-DREGEX=MAE *([-0-9.]*)"
  "-DLOWER_THRESH=0.000"
  "-DUPPER_THRESH=0.005"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-convert-dicom-g" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;plm-convert-dicom-g.mha;--output-ss-list;plm-convert-dicom-g.txt;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-g.cxt"
  )

## -------------------------------------------------------------------------
## plastimatch convert cxt
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-convert-cxt"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c.cxt;--output-ss-img;plm-convert-cxt.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-cxt.txt"
  )
PLM_ADD_TEST (
  "plm-convert-cxt-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-convert-cxt.mha"
  )
ADD_TEST (
  "plm-convert-cxt-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-cxt-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=1.05"
  "-DUPPER_THRESH=1.06"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch convert xio
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-convert-xio"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02;--output-img;chest-phantom-xio.mha;--output-xio;${PLM_TESTING_BUILD_DIR}/plm-convert-xio-xio-output"
  )
PLM_ADD_TEST (
  "plm-convert-xio-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;chest-phantom-xio.mha"
  )
ADD_TEST (
  "plm-convert-xio-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-convert-xio-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-667.5"
  "-DUPPER_THRESH=-666.6"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch usage
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-usage" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  ""
  )

## -------------------------------------------------------------------------
## plastimatch register (group 1)
##   align-center
##   itk-translation
##   itk-rigid
## -------------------------------------------------------------------------
IF (CMAKE_Fortran_COMPILER_WORKS OR F2C_LIBRARY)
  # For Nocedal optimizer
  SET (PLM_BSPLINE_TEST_LOWER_THRESH "-750.0")
  SET (PLM_BSPLINE_TEST_UPPER_THRESH "-748.5")
ELSE (CMAKE_Fortran_COMPILER_WORKS OR F2C_LIBRARY)
  # For Liblbfgs optimizer
  SET (PLM_BSPLINE_TEST_LOWER_THRESH "-706.0")
  SET (PLM_BSPLINE_TEST_UPPER_THRESH "-704.5")
ENDIF (CMAKE_Fortran_COMPILER_WORKS OR F2C_LIBRARY)

PLM_ADD_TEST (
  "plm-register-align-center"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-register-align-center.txt"
  )
PLM_ADD_TEST (
  "plm-register-align-center-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compare;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/plm-register-align-center-img.mha"
  )
ADD_TEST (
  "plm-register-align-center-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-register-align-center-stats.stdout.txt"
  "-DREGEX=MAE *([-0-9.]*)"
  "-DLOWER_THRESH=-0.005"
  "-DUPPER_THRESH=0.005"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-register-itk-translation"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-register-itk-translation.txt"
  )
PLM_ADD_TEST (
  "plm-register-itk-translation-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-register-itk-translation-img.mha"
  )
ADD_TEST (
  "plm-register-itk-translation-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-register-itk-translation-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-637.0"
  "-DUPPER_THRESH=-636.9"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-register-itk-rigid"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-register-itk-rigid.txt"
  )
PLM_ADD_TEST (
  "plm-register-itk-rigid-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-register-itk-rigid-intermediate-img.mha"
  )
ADD_TEST (
  "plm-register-itk-rigid-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-register-itk-rigid-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-965.7"
  "-DUPPER_THRESH=-965.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-register-itk-rigid-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-register-itk-rigid-img.mha"
  )
ADD_TEST (
  "plm-register-itk-rigid-check-2"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-register-itk-rigid-stats-2.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-965.7"
  "-DUPPER_THRESH=-965.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch register (group 2)
##   bspline-single-c
##   bspline-single-h
##   bspline-single-openmp
##   bspline-single-cuda
##   register-bspline-ushort
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-bspline-single-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-c.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-single-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-bspline-single-c-img.mha"
  )
ADD_TEST (
  "plm-bspline-single-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-bspline-single-c-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "-DUPPER_THRESH=${PLM_BSPLINE_TEST_UPPER_THRESH}"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-bspline-single-h" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-h.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-single-h-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-bspline-single-h-img.mha"
  )
ADD_TEST (
  "plm-bspline-single-h-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-bspline-single-h-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "-DUPPER_THRESH=${PLM_BSPLINE_TEST_UPPER_THRESH}"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-bspline-openmp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-openmp.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-openmp-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-bspline-openmp-img.mha"
  )
ADD_TEST (
  "plm-bspline-openmp-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-bspline-openmp-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "-DUPPER_THRESH=${PLM_BSPLINE_TEST_UPPER_THRESH}"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-bspline-cuda" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-cuda.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-bspline-cuda-img.mha"
  )
ADD_TEST (
  "plm-bspline-cuda-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-bspline-cuda-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "-DUPPER_THRESH=${PLM_BSPLINE_TEST_UPPER_THRESH}"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-register-bspline-ushort"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-register-bspline-ushort.txt"
  )
PLM_ADD_TEST (
  "plm-register-bspline-ushort-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-register-bspline-ushort-img.mha"
  )
ADD_TEST (
  "plm-register-bspline-ushort-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-register-bspline-ushort-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=107.6"
  "-DUPPER_THRESH=108.6"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )


## -------------------------------------------------------------------------
## plastimatch register (group 3) - rect 2 & rect 3
##   register-rect-bspline
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-register-rect-bspline" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-register-rect-bspline.txt"
  )

## -------------------------------------------------------------------------
## plastimatch compose
##   This test uses registers from gauss-1 to gauss-2 which are done above, 
##   and composes with results from an registration from gauss-2 to 
##   gauss-5.  The composed vf should be correctly align gauss-1 to gauss-5.
##   
##   plm-register-compose    Make vf for testing composition
##   plm-compose-a           Compose translation and vector field
##   plm-compose-b           Compose vector field and vector field
##   plm-compose-c           Compose bspline and vector field
##   
##   Remark: these tests are correct, in the sense that they work as 
##   implemented.  Transformation vectors which are not defined are 
##   set to zero in plastimatch, and this causes artifacts in the composed 
##   vector field.
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-register-compose" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-register-compose.txt"
  )
PLM_ADD_TEST (
  "plm-compose-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compose;plm-register-itk-translation-xf.txt;plm-register-compose-vf.mha;plm-compose-a-vf.mha"
  )
PLM_ADD_TEST (
  "plm-compose-a-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-compose-a-vf.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-compose-a-warp.mha"
  )
PLM_ADD_TEST (
  "plm-compose-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-compose-a-vf.mha"
  )
ADD_TEST (
  "plm-compose-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-compose-a-stats.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=13.1"
  "-DUPPER_THRESH=13.3"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-compose-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compose;plm-register-itk-translation-vf.mha;plm-register-compose-vf.mha;plm-compose-b-vf.mha"
  )
PLM_ADD_TEST (
  "plm-compose-b-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-compose-b-vf.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-compose-b-warp.mha"
  )
PLM_ADD_TEST (
  "plm-compose-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-compose-b-vf.mha"
  )
ADD_TEST (
  "plm-compose-b-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-compose-b-stats.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=13.1"
  "-DUPPER_THRESH=13.3"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-compose-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compose;plm-bspline-single-h-xf.txt;plm-register-compose-vf.mha;plm-compose-c-vf.mha"
  )
PLM_ADD_TEST (
  "plm-compose-c-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-compose-c-vf.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-compose-c-warp.mha"
  )
PLM_ADD_TEST (
  "plm-compose-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-compose-c-vf.mha"
  )
ADD_TEST (
  "plm-compose-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-compose-c-stats.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=12.4"
  "-DUPPER_THRESH=12.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch resample
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-resample-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "resample;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-resample-a.mha;--input-type;float;--output-type;float;--subsample;2 2 2"
  )

## -------------------------------------------------------------------------
## plastimatch warp pointset
##   plm warp pointset a:  xf = bspline, output fcsv
##   plm warp pointset b:  xf = bspline, output txt
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-warp-pointset-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/plm-register-rect-bspline-xf.txt;--input;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;--output-pointset;${PLM_TESTING_BUILD_DIR}/plm-warp-pointset-a.fcsv"
  )
PLM_ADD_TEST (
  "plm-warp-pointset-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/plm-register-rect-bspline-xf.txt;--input;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;--output-pointset;${PLM_TESTING_BUILD_DIR}/plm-warp-pointset-b.txt"
  )

## -------------------------------------------------------------------------
## plastimatch warp
##   plm warp a: xf = bspline, algorithm = itk
##   plm warp b: xf = bspline, algorithm = native
##   plm warp c: xf = bspline, algorithm = itk, interp = nn
##   plm warp d: xf = bspline, algorithm = native, interp = nn
##   plm warp e: input = xio, multiple outputs
##   plm warp f: input = dicom, multiple outputs
##   plm warp g: xf = translation
##   plm warp h: xf = translation, ushort images
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-a-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-a.mha"
  )
PLM_ADD_TEST (
  "plm-warp-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-a-vf.mha"
  )
ADD_TEST (
  "plm-warp-a-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-a-stats.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=5.77"
  "-DUPPER_THRESH=5.78"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-b-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-b.mha"
  )
PLM_ADD_TEST (
  "plm-warp-b-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-b-vf.mha"
  )
ADD_TEST (
  "plm-warp-b-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-b-stats-1.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=5.77"
  "-DUPPER_THRESH=5.78"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-warp-b-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-b.mha"
  )
ADD_TEST (
  "plm-warp-b-check-2"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-b-stats-2.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-775.7"
  "-DUPPER_THRESH=-775.6"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-c-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/sphere-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-c.mha;--interpolation;nn"
  )
PLM_ADD_TEST (
  "plm-warp-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-c.mha"
  )
ADD_TEST (
  "plm-warp-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-c-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=9.2"
  "-DUPPER_THRESH=9.3"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-d" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-d-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/sphere-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-d.mha;--interpolation;nn"
  )
PLM_ADD_TEST (
  "plm-warp-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-d.mha"
  )
ADD_TEST (
  "plm-warp-d-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-d-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=9.2"
  "-DUPPER_THRESH=9.3"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-e" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-bspline-chest-phantom.txt;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02;--output-xio;${PLM_TESTING_BUILD_DIR}/plm-warp-e-xio;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-img.nrrd;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-list.txt;--output-labelmap;${PLM_TESTING_BUILD_DIR}/plm-warp-e-labelmap.nrrd;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-e-img.nrrd;--output-prefix;${PLM_TESTING_BUILD_DIR}/plm-warp-e;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-warp-e.cxt;--output-colormap;${PLM_TESTING_BUILD_DIR}/plm-warp-e.ctbl"
  )
PLM_ADD_TEST (
  "plm-warp-e-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e-img.nrrd"
  )
ADD_TEST (
  "plm-warp-e-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-661.5"
  "-DUPPER_THRESH=-661.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-warp-e-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e_Patient.mha"
  )
ADD_TEST (
  "plm-warp-e-check-2"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-2.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=0.385"
  "-DUPPER_THRESH=0.395"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-warp-e-stats-3"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e-labelmap.nrrd"
  )
ADD_TEST (
  "plm-warp-e-check-3"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-3.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=0.67"
  "-DUPPER_THRESH=0.68"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-warp-e-stats-4"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e-ss-img.nrrd"
  )
ADD_TEST (
  "plm-warp-e-check-4"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-4.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=1.05"
  "-DUPPER_THRESH=1.15"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
ADD_TEST (
  "plm-warp-e-check-5"
  ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/plm-warp-e-ss-list.txt"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-list.txt")
ADD_TEST (
  "plm-warp-e-check-6"
  ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/plm-warp-e.ctbl"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e.ctbl")

PLM_ADD_TEST (
  "plm-warp-f" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-bspline-chest-phantom.txt;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-warp-f-dicom;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-warp-f-ss-img.nrrd;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-warp-f-ss-list.txt;--output-labelmap;${PLM_TESTING_BUILD_DIR}/plm-warp-f-labelmap.nrrd;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-f-img.nrrd;--output-prefix;${PLM_TESTING_BUILD_DIR}/plm-warp-f;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-warp-f.cxt"
  )
PLM_ADD_TEST (
  "plm-warp-f-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-f-img.nrrd"
  )
ADD_TEST (
  "plm-warp-f-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-f-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-661.5"
  "-DUPPER_THRESH=-661.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-g" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-translation-1.txt;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-g-img.nrrd"
  )
PLM_ADD_TEST (
  "plm-warp-g-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-g-img.nrrd"
  )
ADD_TEST (
  "plm-warp-g-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-g-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-675.5"
  "-DUPPER_THRESH=-674.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-h" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-translation-1.txt;--input;${PLM_TESTING_BUILD_DIR}/gauss-ushort-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-h-img.nrrd"
  )
PLM_ADD_TEST (
  "plm-warp-h-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-h-img.nrrd"
  )
ADD_TEST (
  "plm-warp-h-check-1"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-h-stats-1.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=113.5"
  "-DUPPER_THRESH=114.0"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plm-api - Tests of plastimatch API
##   plm drr api a:
##   plm registration api a:
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-drr-api-a"
  ""
  ${PLM_PLASTIMATCH_TESTING_PATH}/plm_drr_api_a
  ""
  )
PLM_ADD_TEST (
  "plm-registration-api-a"
  ""
  ${PLM_PLASTIMATCH_TESTING_PATH}/plm_registration_api_a
  ""
  )

## -------------------------------------------------------------------------
## bragg-curve
## proton-dose
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "bragg-curve"
  ""
  ${PLM_PLASTIMATCH_PATH}/bragg_curve
  "-O;${PLM_TESTING_BUILD_DIR}/bragg-curve.txt"
  )
ADD_TEST (bragg-curve-check ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/bragg-curve.txt"
  "${PLM_TESTING_BUILD_DIR}/bragg-curve.txt")

PLM_ADD_TEST (
  "proton-dose"
  ""
  ${PLM_PLASTIMATCH_PATH}/proton_dose
  "-src;1 0 0;-iso;0 0 0;-vup;0 0 1;-u;1;-p;${PLM_TESTING_DATA_DIR}/bragg-curve.txt;--debug;${PLM_TESTING_BUILD_DIR}/rect-1.mha;${PLM_TESTING_BUILD_DIR}/proton-dose.mha"
  )

## -------------------------------------------------------------------------
## xf_to_xf
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "xf-to-xf-1" 
  ""
  ${PLM_PLASTIMATCH_PATH}/xf_to_xf
  "--input;${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;--output;${PLM_TESTING_BUILD_DIR}/xf-to-xf-1.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )

PLM_ADD_TEST (
  "xf-to-xf-2" 
  ""
  ${PLM_PLASTIMATCH_PATH}/xf_to_xf
  "--input;${PLM_TESTING_BUILD_DIR}/xf-to-xf-1.mha;--output;${PLM_TESTING_BUILD_DIR}/xf-to-xf-2.txt;--output-type;bspline;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;--grid-spacing;10 10 10"
  )

PLM_ADD_TEST (
  "xf-to-xf-3" 
  ""
  ${PLM_PLASTIMATCH_PATH}/xf_to_xf
  "--input;${PLM_TESTING_BUILD_DIR}/xf-to-xf-2.txt;--output;${PLM_TESTING_BUILD_DIR}/xf-to-xf-3.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )


## -------------------------------------------------------------------------
## slicer tests
## -------------------------------------------------------------------------
IF (Slicer3_DIR)
#  slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)
ENDIF (Slicer3_DIR)
