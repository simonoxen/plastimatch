
IF (NOT CUDA_FOUND)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "fdk-cuda"
    "plastimatch-bspline-cuda" 
    )
ENDIF (NOT CUDA_FOUND)
