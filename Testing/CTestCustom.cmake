
SET (CUDA_FOUND @CUDA_FOUND@)
SET (BROOK_FOUND @BROOK_FOUND@)

#SET (REDUCED_TEST ON)
SET (REDUCED_TEST OFF)

IF (NOT CUDA_FOUND)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "fdk-cuda"
    "plastimatch-bspline-cuda" 
    )
ENDIF (NOT CUDA_FOUND)

IF (NOT BROOK_FOUND)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "fdk-brook"
    )
ENDIF (NOT BROOK_FOUND)

IF (REDUCED_TEST)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "synth-test-1"
    "synth-test-2"
    "bspline-bxf"
    "bspline-bxf-check"
    "bspline-a"
    "bspline-b"
    "bspline-c"
    "bspline-c-check"
    "bspline-d"
    "bspline-e"
    "bspline-f"
    "bspline-f-check"
    "drr"
    "fdk-cpu"
    "fdk-cuda"
    "plastimatch-itk-translation"
    "plastimatch-bspline-single-c" 
    "plastimatch-bspline-single-f" 
    "plastimatch-bspline-openmp" 
    "xf-to-xf"
    )
ENDIF (REDUCED_TEST)
