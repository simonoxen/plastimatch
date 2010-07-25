PROJECT(PLM_Testing)

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

IF (PLM_TEST_DICOM)
  PLM_DOWNLOAD_AND_EXTRACT (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/934/chest-phantom-dicomrt-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")
ENDIF (PLM_TEST_DICOM)

IF (PLM_TEST_XIO)
  PLM_DOWNLOAD_AND_EXTRACT (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/939/chest-phantom-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")
ENDIF (PLM_TEST_XIO)

SET (SYNTH_MHA_SIZE 38)

## -------------------------------------------------------------------------
## synth-test - create synthetic mha files
##   synth-1.mha   Centered gauss
##   synth-2.mha   Off-center gauss
##   synth-3.mha   Uchar sphere, bg=0, fg=255
##   synth-4.mha   Inverted gauss
##   synth-5.mha   Rect
##   synth-6.mha   Inverged rect
##   synth-7.mha   Gauss with inverted pixel spacing
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "synth-1"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth-1.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "synth-2"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth-2.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "synth-3"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth-3.mha;--output-type;uchar;--pattern;sphere;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;255;--sphere-center;0 0 0;--sphere-radius;10 10 10"
  )

PLM_ADD_TEST (
  "synth-4"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth-4.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;-1000;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "synth-5"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;rect-1.mha;--output-type;float;--pattern;rect;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;0;--foreground;-1000"
  )

PLM_ADD_TEST (
  "synth-6"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;rect-2.mha;--output-type;float;--pattern;rect;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--rect-size;-30 70 -50 50 -50 50"
  )

PLM_ADD_TEST (
  "synth-7"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth-7.mha;--output-type;float;--pattern;gauss;--origin;+25 +25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--spacing;-1.31579 -1.31579 2;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )

## -------------------------------------------------------------------------
## bspline-bxf - read and write bxf format
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "bspline-bxf"
  ""
  ${PLM_PLASTIMATCH_TESTING_PATH_HACK}/bspline_test_1
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt"
  )

ADD_EXECUTABLE(bspline_test_1 bspline_test_1.cxx)
IF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
  SET_TARGET_PROPERTIES(bspline_test_1 PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
ENDIF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
TARGET_LINK_LIBRARIES(bspline_test_1 ${GPUIT_LIBRARIES})

ADD_TEST (bspline-bxf-check ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt"
  "${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt")

## -------------------------------------------------------------------------
## bspline - bspline algorithm flavors
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "bspline-a"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;a;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;b;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;c;-x;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;d;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;e;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;f;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;g;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;h;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;c;-M;mi;-O;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1.mha;${PLM_TESTING_BUILD_DIR}/synth-1.mha;${PLM_TESTING_BUILD_DIR}/synth-4.mha"
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
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
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
## drr
##   drr-a   synth-1.mha (normal image), exact
##   drr-b   synth-1.mha (normal image), uniform
##   drr-c   synth-7.mha (negative spacing), exact
##   drr-d   synth-7.mha (negative spacing), uniform
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "drr-a"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-a;20;-O;drr-a/out_;${PLM_TESTING_BUILD_DIR}/synth-1.mha"
  )
PLM_ADD_TEST (
  "drr-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-a;20;-O;drr-b/out_;-i;uniform;${PLM_TESTING_BUILD_DIR}/synth-1.mha"
  )
PLM_ADD_TEST (
  "drr-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-a;20;-O;drr-c/out_;${PLM_TESTING_BUILD_DIR}/synth-7.mha"
  )
PLM_ADD_TEST (
  "drr-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-a;20;-O;drr-d/out_;-i;uniform;${PLM_TESTING_BUILD_DIR}/synth-7.mha"
  )
PLM_ADD_TEST (
  "drr-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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

## -------------------------------------------------------------------------
## drr-cuda
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "drr-cuda"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-A;cuda;-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-cuda/out_;${PLM_TESTING_BUILD_DIR}/synth-1.mha"
  )
PLM_ADD_TEST (
  "drr-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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

## -------------------------------------------------------------------------
## fdk cpu
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-cpu"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-a;0 19;-O;fdk-cpu.mha"
  )
PLM_ADD_TEST (
  "fdk-cpu-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;fdk-cpu.mha"
  )
ADD_TEST (
  "fdk-cpu-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-cpu-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=-987.5"
  "-DUPPER_THRESH=-986.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## fdk brook
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-brook"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;brook;-a;0 19;-O;fdk_brook_output.mha"
  )
PLM_ADD_TEST (
  "fdk-brook-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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

## -------------------------------------------------------------------------
## fdk cuda
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-cuda"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;cuda;-a;0 19;-O;fdk_cuda_output.mha"
  )
PLM_ADD_TEST (
  "fdk-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;fdk_cuda_output.mha"
  )
ADD_TEST (
  "fdk-cuda-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/fdk-cuda-stats.stdout.txt"
  "-DREGEX=MAX *([-0-9.]*)"
  "-DLOWER_THRESH=-987.5"
  "-DUPPER_THRESH=-986.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## landmark_warp
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "landmark-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/landmark_warp
  "-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-vf.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp.mha;${PLM_TESTING_DATA_DIR}/xf-tps-1.txt;${PLM_TESTING_BUILD_DIR}/synth-2.mha"
  )

## -------------------------------------------------------------------------
## plastimatch convert dicom (and dicom rt)
##   plm convert dicom a: ct only
##   plm convert dicom b: rtss only, with dicom-dir reference
##   plm convert dicom c: rtss only, with fixed image reference
##   plm convert dicom d: dose only, native sampling rate
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-convert-dicom-a"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output;plm-convert-dicom-a.mha"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;plm-convert-dicom-b.mha;--dicom-dir;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;plm-convert-dicom-c.mha;--fixed;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a.mha;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c.cxt"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/DOSE.20080627A.TRAINING4FLD.dcm;--output-dose-img;plm-convert-dicom-d.mha"
  )
PLM_ADD_TEST (
  "plm-convert-dicom-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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

## -------------------------------------------------------------------------
## plastimatch convert cxt
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-convert-cxt"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c.cxt;--output-ss-img;plm-convert-cxt.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-cxt.txt"
  )
PLM_ADD_TEST (
  "plm-convert-cxt-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02;--output-img;chest-phantom-xio.mha;--output-xio;${PLM_TESTING_BUILD_DIR}/plm-convert-xio-xio-output"
  )
PLM_ADD_TEST (
  "plm-convert-xio-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  ""
  )

## -------------------------------------------------------------------------
## plastimatch register
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
  "plm-itk-translation"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-itk-translation.txt"
  )
PLM_ADD_TEST (
  "plm-itk-translation-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;plm-itk-translation-img.mha"
  )
ADD_TEST (
  "plm-itk-translation-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-itk-translation-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-637.0"
  "-DUPPER_THRESH=-636.9"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-bspline-single-c" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-c.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-single-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-h.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-single-h-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-openmp.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-openmp-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-cuda.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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

## -------------------------------------------------------------------------
## plastimatch resample
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-resample-a" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "resample;--input;${PLM_TESTING_BUILD_DIR}/synth-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-resample-a.mha;--input-type;float;--output-type;float;--subsample;2 2 2"
  )

## -------------------------------------------------------------------------
## plastimatch warp
##   plm warp a: xf = bspline, algorithm = itk
##   plm warp b: xf = bspline, algorithm = native
##   plm warp c: xf = bspline, algorithm = itk, interp = nn
##   plm warp d: xf = bspline, algorithm = native, interp = nn
##   plm warp e: input = xio, multiple outputs
##   plm warp f: input = dicom, multiple outputs
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-a-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-a.mha"
  )
PLM_ADD_TEST (
  "plm-warp-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-b-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-b.mha"
  )
PLM_ADD_TEST (
  "plm-warp-b-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-c-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth-3.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-c.mha;--interpolation;nn"
  )
PLM_ADD_TEST (
  "plm-warp-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-d-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth-3.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-d.mha;--interpolation;nn"
  )
PLM_ADD_TEST (
  "plm-warp-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-bspline-chest-phantom.txt;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02;--output-xio;${PLM_TESTING_BUILD_DIR}/plm-warp-e-xio;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-img.nrrd;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-list.txt;--output-labelmap;${PLM_TESTING_BUILD_DIR}/plm-warp-e-labelmap.nrrd;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-e-img.nrrd;--output-prefix;${PLM_TESTING_BUILD_DIR}/plm-warp-e;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-warp-e.cxt"
  )
PLM_ADD_TEST (
  "plm-warp-e-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;plm-warp-e-labelmap.nrrd"
  )
ADD_TEST (
  "plm-warp-e-check-3"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-3.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=0.68"
  "-DUPPER_THRESH=0.69"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )
PLM_ADD_TEST (
  "plm-warp-e-stats-4"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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


PLM_ADD_TEST (
  "plm-warp-f" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-bspline-chest-phantom.txt;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-warp-f-dicom;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-warp-f-ss-img.nrrd;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-warp-f-ss-list.txt;--output-labelmap;${PLM_TESTING_BUILD_DIR}/plm-warp-f-labelmap.nrrd;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-f-img.nrrd;--output-prefix;${PLM_TESTING_BUILD_DIR}/plm-warp-f;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-warp-f.cxt"
  )
PLM_ADD_TEST (
  "plm-warp-f-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
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

## -------------------------------------------------------------------------
## xf_to_xf
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "xf-to-xf-1" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/xf_to_xf
  "--input;${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;--output;${PLM_TESTING_BUILD_DIR}/xf-to-xf-1.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )

PLM_ADD_TEST (
  "xf-to-xf-2" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/xf_to_xf
  "--input;${PLM_TESTING_BUILD_DIR}/xf-to-xf-1.mha;--output;${PLM_TESTING_BUILD_DIR}/xf-to-xf-2.txt;--output-type;bspline;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;--grid-spacing;10 10 10"
  )

PLM_ADD_TEST (
  "xf-to-xf-3" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/xf_to_xf
  "--input;${PLM_TESTING_BUILD_DIR}/xf-to-xf-2.txt;--output;${PLM_TESTING_BUILD_DIR}/xf-to-xf-3.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )


## -------------------------------------------------------------------------
## slicer tests
## -------------------------------------------------------------------------
IF (Slicer3_DIR)
#  slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)
ENDIF (Slicer3_DIR)
