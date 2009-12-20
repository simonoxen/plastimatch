PROJECT(PLM_Testing)

SET (SYNTH_MHA_SIZE 38)

## -------------------------------------------------------------------------
## synth-test - create synthetic mha files
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "synth-test-1"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth_1.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "synth-test-2"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth_2.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

PLM_ADD_TEST (
  "synth-test-3"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth_3.mha;--output-type;uchar;--pattern;sphere;--origin;-25 -25 -25;--resolution;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;255;--sphere-center;0 0 0;--sphere-radius;10 10 10"
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
  "-a;steepest;-m;10;-f;a;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-a-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-a.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=18945.5"
  "-DUPPER_THRESH=18945.7"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-b"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;b;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-b-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-b.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=18945.5"
  "-DUPPER_THRESH=18945.7"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-c"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;c;-x;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-c-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-c.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=7429.0"
  "-DUPPER_THRESH=7429.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-d"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;d;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-d-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-d.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=7429.0"
  "-DUPPER_THRESH=7429.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-e"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;e;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-e-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-e.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=7429.0"
  "-DUPPER_THRESH=7429.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-f"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;f;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-f-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-f.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=7429.0"
  "-DUPPER_THRESH=7429.4"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-g"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;g;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-g-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-g.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=7428.5"
  "-DUPPER_THRESH=7428.9"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "bspline-h"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;h;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )
ADD_TEST (bspline-h-check
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/bspline-h.stdout.txt"
  "-DREGEX=^MSE\\\\[  10] *([0-9.]*)"
  "-DLOWER_THRESH=7428.5"
  "-DUPPER_THRESH=7428.9"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## drr
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "drr"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-a;20;${PLM_TESTING_BUILD_DIR}/synth_1.mha"
  )
PLM_ADD_TEST (
  "drr-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;out_0010.pfm"
  )
ADD_TEST (
  "drr-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/drr-stats.stdout.txt"
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
  "-A;cuda;-a;20;-O;cuda_drr_;${PLM_TESTING_BUILD_DIR}/synth_1.mha"
  )
PLM_ADD_TEST (
  "drr-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;cuda_drr_0010.pfm"
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
  "-I;${PLM_TESTING_BUILD_DIR};-f;none;-a;0 19;-O;fdk_cpu_output.mha"
  )
PLM_ADD_TEST (
  "fdk-cpu-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;fdk_cpu_output.mha"
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
  "-I;${PLM_TESTING_BUILD_DIR};-f;none;-A;brook;-a;0 19;-O;fdk_brook_output.mha"
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
  "-I;${PLM_TESTING_BUILD_DIR};-f;none;-A;cuda;-a;0 19;-O;fdk_cuda_output.mha"
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
  "-DLOWER_THRESH=-750.0"
  "-DUPPER_THRESH=-748.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-bspline-single-f" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-f.txt"
  )
PLM_ADD_TEST (
  "plm-bspline-single-f-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;plm-bspline-single-f-img.mha"
  )
ADD_TEST (
  "plm-bspline-single-f-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-bspline-single-f-stats.stdout.txt"
  "-DREGEX=AVE *([-0-9.]*)"
  "-DLOWER_THRESH=-750.0"
  "-DUPPER_THRESH=-748.5"
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
  "-DLOWER_THRESH=-750.0"
  "-DUPPER_THRESH=-748.5"
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
  "-DLOWER_THRESH=-750.0"
  "-DUPPER_THRESH=-748.5"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch resample
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-resample-a" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "resample;--input;${PLM_TESTING_BUILD_DIR}/synth_1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-resample-a.mha;--input-type;float;--output-type;float;--subsample;2 2 2"
  )

## -------------------------------------------------------------------------
## plastimatch warp
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plm-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-a-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth_1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-a.mha"
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
  "-DLOWER_THRESH=2.26"
  "-DUPPER_THRESH=2.28"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-b" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-b-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth_1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-b.mha"
  )
PLM_ADD_TEST (
  "plm-warp-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;plm-warp-b-vf.mha"
  )
ADD_TEST (
  "plm-warp-b-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-b-stats.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=2.26"
  "-DUPPER_THRESH=2.28"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

PLM_ADD_TEST (
  "plm-warp-c" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-c-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth_3.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-warp-c.mha;--interpolation;nn"
  )
PLM_ADD_TEST (
  "plm-warp-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "stats;--input;plm-warp-c-vf.mha"
  )
ADD_TEST (
  "plm-warp-c-check"
  ${CMAKE_COMMAND} 
  "-DINFILE=${PLM_TESTING_BUILD_DIR}/plm-warp-c-stats.stdout.txt"
  "-DREGEX=Mean: *([-0-9.]*)"
  "-DLOWER_THRESH=2.26"
  "-DUPPER_THRESH=2.28"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## tps_warp
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "tps-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/tps_warp
  "-V;${PLM_TESTING_BUILD_DIR}/tps-warp-vf.mha;-O;${PLM_TESTING_BUILD_DIR}/tps-warp.mha;${PLM_TESTING_DATA_DIR}/xf-tps-1.txt;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )


## -------------------------------------------------------------------------
## xf_to_xf
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "xf-to-xf" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/xf_to_xf
  "--input;${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;--output;${PLM_TESTING_BUILD_DIR}/vf-bspline-1.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )


## -------------------------------------------------------------------------
## slicer tests
## -------------------------------------------------------------------------
IF (Slicer3_DIR)
#  slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)
ENDIF (Slicer3_DIR)

