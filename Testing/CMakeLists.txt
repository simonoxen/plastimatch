PROJECT(PLM_Testing)

## synth_test_1 - create synthetic mha file
PLM_ADD_TEST (
  "synth_test_1"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth_1.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;50 50 50;--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )
## synth_test_2 - create synthetic mha file
PLM_ADD_TEST (
  "synth_test_2"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/synthetic_mha
  "--output;synth_1.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;50 50 50;--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )

## bspline_test_1 - read and write bxf
## bspline_test_1_compare - compare written bxf with reference
PLM_ADD_TEST (
  "bspline_test_1"
  ""
  ${PLM_PLASTIMATCH_TESTING_PATH_HACK}/bspline_test_1
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt"
  )
ADD_EXECUTABLE(bspline_test_1 bspline_test_1.cxx)
TARGET_LINK_LIBRARIES(bspline_test_1 ${GPUIT_LIBRARIES})
ADD_TEST (bspline_test_1_compare ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt"
  "${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt")

## plastimatch_usage_1
## make sure that plastimatch runs
PLM_ADD_TEST (
  "plastimatch_usage_1" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  ""
  )

## ??
if (0)
  PLM_ADD_TEST (
    "plastimatch_y1" 
    "C:/gcs6/idata/synth_mse"
    ${PLM_PLASTIMATCH_TESTING_PATH_HACK}/plastimatch
    "parms.txt"
    )
endif (0)

