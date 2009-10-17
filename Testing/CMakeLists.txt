PROJECT(PLM_Testing)

SET (SYNTH_MHA_SIZE 38)

## synth-test - create synthetic mha files
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

## bspline-bxf - read and write bxf
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

ADD_TEST (bspline-bxf-compare ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt"
  "${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt")

## bspline - bspline driver program
PLM_ADD_TEST (
  "bspline-a"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;a;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )

PLM_ADD_TEST (
  "bspline-b"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;b;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )

PLM_ADD_TEST (
  "bspline-c"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;c;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )

PLM_ADD_TEST (
  "bspline-d"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;d;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )

PLM_ADD_TEST (
  "bspline-e"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;e;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )

PLM_ADD_TEST (
  "bspline-f"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/bspline
  "-a;steepest;-m;10;-f;f;${PLM_TESTING_BUILD_DIR}/synth_1.mha;${PLM_TESTING_BUILD_DIR}/synth_2.mha"
  )

## drr
PLM_ADD_TEST (
  "drr"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/drr_c
  "-a;20;-e;${PLM_TESTING_BUILD_DIR}/synth_1.mha"
  )

## fdk
PLM_ADD_TEST (
  "fdk-cpu"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk_c
  "-I;${PLM_TESTING_BUILD_DIR};-a;0 19;-O;fdk_cpu_output.mha"
  )

PLM_ADD_TEST (
  "fdk-cuda"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/fdk_cuda
  "-I;${PLM_TESTING_BUILD_DIR};-a;0 19;-O;fdk_cuda_output.mha"
  )

## plastimatch
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

## xf_to_xf
PLM_ADD_TEST (
  "xf-to-xf" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/xf_to_xf
  "--input;${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;--output;${PLM_TESTING_BUILD_DIR}/vf-bspline-1.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )
