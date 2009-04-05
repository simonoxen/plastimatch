PROJECT(PLM_Testing)

## bspline_test_1 - read and write bxf
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

