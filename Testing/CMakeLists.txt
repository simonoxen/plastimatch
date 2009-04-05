PROJECT(PLM_Testing)

PLM_ADD_TEST (
  "bspline_1_test"
  ""
  ${PLM_PLASTIMATCH_TESTING_PATH_HACK}/bspline_1_test
  "${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt"
  )

ADD_EXECUTABLE(bspline_1_test bspline_1.cxx)
