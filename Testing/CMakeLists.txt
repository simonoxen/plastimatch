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
  "--output;synth_2.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--resolution;50 50 50;--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
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
IF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
  SET_TARGET_PROPERTIES(bspline_test_1 PROPERTIES LINK_FLAGS ${OPENMP_LDFLAGS})
ENDIF(NOT ${OPENMP_LDFLAGS} STREQUAL "")
TARGET_LINK_LIBRARIES(bspline_test_1 ${GPUIT_LIBRARIES})
ADD_TEST (bspline_test_1_compare ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt"
  "${PLM_TESTING_BUILD_DIR}/xf-bspline-1.txt")

## plastimatch
PLM_ADD_TEST (
  "plastimatch_usage" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  ""
  )

PLM_ADD_TEST (
  "plastimatch-register-itk-translation"
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-register-itk-translation.txt"
  )

PLM_ADD_TEST (
  "plastimatch-register-plm-bspline-single" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-register-plm-bspline-single.txt"
  )

PLM_ADD_TEST (
  "plastimatch-register-plm-bspline-openmp" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-register-plm-bspline-openmp.txt"
  )

PLM_ADD_TEST (
  "plastimatch-register-plm-bspline-cuda" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plastimatch-register-plm-bspline-cuda.txt"
  )

## xf_to_xf
PLM_ADD_TEST (
  "xf_to_xf_test_1" 
  ""
  ${PLM_PLASTIMATCH_PATH_HACK}/xf_to_xf
  "--input;${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;--output;${PLM_TESTING_BUILD_DIR}/vf-bspline-1.mha;--output-type;vf;--dims;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )
