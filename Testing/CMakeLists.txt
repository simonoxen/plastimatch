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

## -------------------------------------------------------------------------
## drr
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "drr"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/drr
  "-a;20;${PLM_TESTING_BUILD_DIR}/synth_1.mha"
  )

## -------------------------------------------------------------------------
## fdk cpu
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-cpu"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk
  "-I;${PLM_TESTING_BUILD_DIR};-a;0 19;-O;fdk_cpu_output.mha"
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
  "-DLOWER_THRESH=-999.9"
  "-DUPPER_THRESH=-999.8"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## fdk brook
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-brook"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk
  "-I;${PLM_TESTING_BUILD_DIR};-A;brook;-a;0 19;-O;fdk_brook_output.mha"
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
  "-DLOWER_THRESH=-999.9"
  "-DUPPER_THRESH=-999.8"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## fdk cuda
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "fdk-cuda"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk
  "-I;${PLM_TESTING_BUILD_DIR};-A;cuda;-a;0 19;-O;fdk_cuda_output.mha"
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
  "-DLOWER_THRESH=-999.9"
  "-DUPPER_THRESH=-999.8"
  -P ${PLM_TESTING_SOURCE_DIR}/PLM_CTest_Check.cmake
  )

## -------------------------------------------------------------------------
## plastimatch register
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plastimatch-usage" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  ""
  )

PLM_ADD_TEST (
  "plastimatch-itk-translation"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-itk-translation.txt"
  )

PLM_ADD_TEST (
  "plastimatch-bspline-single-c" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-bspline-single-c.txt"
  )

PLM_ADD_TEST (
  "plastimatch-bspline-single-f" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-bspline-single-f.txt"
  )

PLM_ADD_TEST (
  "plastimatch-bspline-openmp" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-bspline-openmp.txt"
  )

PLM_ADD_TEST (
  "plastimatch-bspline-cuda" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-bspline-cuda.txt"
  )

## -------------------------------------------------------------------------
## plastimatch warp
## -------------------------------------------------------------------------
PLM_ADD_TEST (
  "plastimatch-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plastimatch-warp-a-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth_1.mha;--output;${PLM_TESTING_BUILD_DIR}/plastimatch-warp-a.mha"
  )

PLM_ADD_TEST (
  "plastimatch-warp-b" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plastimatch-warp-b-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/synth_1.mha;--output;${PLM_TESTING_BUILD_DIR}/plastimatch-warp-b.mha"
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

