
SET (CUDA_FOUND @CUDA_FOUND@)
SET (BROOK_FOUND @BROOK_FOUND@)
SET (PLM_TEST_BSPLINE_FLAVORS @PLM_TEST_BSPLINE_FLAVORS@)
SET (PLM_TEST_DICOM @PLM_TEST_DICOM@)
SET (PLM_TESTING_BUILD_DIR "@PLM_TESTING_BUILD_DIR@")
SET (PLM_PLASTIMATCH_PATH_HACK "@PLM_PLASTIMATCH_PATH_HACK@")

#SET (REDUCED_TEST ON)
SET (REDUCED_TEST OFF)

## If we don't have functioning GPU, don't run cuda tests
SET (RUN_CUDA_TESTS OFF)
IF (CUDA_FOUND)
  EXECUTE_PROCESS (COMMAND 
#    "${PLM_PLASTIMATCH_PATH_HACK}/cuda_probe.exe"
    "${PLM_PLASTIMATCH_PATH_HACK}/cuda_probe"
    RESULT_VARIABLE CUDA_PROBE_RESULT
    OUTPUT_VARIABLE CUDA_PROBE_STDOUT
    ERROR_VARIABLE CUDA_PROBE_STDERR
    )
  FILE (WRITE "${PLM_TESTING_BUILD_DIR}/cuda_probe_result.txt"
    "${CUDA_PROBE_RESULT}")
  FILE (WRITE "${PLM_TESTING_BUILD_DIR}/cuda_probe_stdout.txt"
    "${CUDA_PROBE_STDOUT}")
  FILE (WRITE "${PLM_TESTING_BUILD_DIR}/cuda_probe_stderr.txt"
    "${CUDA_PROBE_STDERR}")
  STRING (REGEX MATCH "NOT cuda capable" CUDA_PROBE_NOT_CAPABLE
    "${CUDA_PROBE_STDOUT}")
  IF (NOT CUDA_PROBE_NOT_CAPABLE)
    SET (RUN_CUDA_TESTS ON)
  ENDIF (NOT CUDA_PROBE_NOT_CAPABLE)
ENDIF (CUDA_FOUND)

## If we didn't get dicom test data, don't run dicom tests
IF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "plm-convert-dicom"
    "plm-convert-dicom-stats"
    "plm-convert-dicom-check"
    "plm-convert-dicom-rtss-a"
    "plm-convert-dicom-rtss-stats-a"
    "plm-convert-dicom-rtss-check-a"
    "plm-convert-dicom-rtss-b"
    "plm-convert-dicom-rtss-stats-b"
    "plm-convert-dicom-rtss-check-b"
    )
ENDIF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")

IF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "plm-convert-xio"
    "plm-convert-xio-stats"
    "plm-convert-xio-check"
    )
ENDIF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")

IF (NOT RUN_CUDA_TESTS)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "drr-cuda"
    "drr-cuda-stats"
    "drr-cuda-check"
    "fdk-cuda"
    "fdk-cuda-stats"
    "fdk-cuda-check"
    "plm-bspline-cuda" 
    "plm-bspline-cuda-stats" 
    "plm-bspline-cuda-check" 
    )
ENDIF (NOT RUN_CUDA_TESTS)

IF (NOT BROOK_FOUND)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "fdk-brook"
    "fdk-brook-stats"
    "fdk-brook-check"
    )
ENDIF (NOT BROOK_FOUND)

## Don't test unused algorithms
IF (NOT PLM_TEST_BSPLINE_FLAVORS)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "bspline-a"
    "bspline-a-check"
    "bspline-b"
    "bspline-b-check"
    "bspline-d"
    "bspline-d-check"
    "bspline-e"
    "bspline-e-check"
    "bspline-f"
    "bspline-f-check"
    )
ENDIF (NOT PLM_TEST_BSPLINE_FLAVORS)

## Don't delete from the list, comment out instead.
IF (REDUCED_TEST)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "synth-test-1"
    "synth-test-2"
    "synth-test-3"
    "bspline-bxf"
    "bspline-bxf-check"
    "bspline-a"
    "bspline-a-check"
    "bspline-b"
    "bspline-b-check"
    "bspline-c"
    "bspline-c-check"
    "bspline-d"
    "bspline-d-check"
    "bspline-e"
    "bspline-e-check"
    "bspline-f"
    "bspline-f-check"
    "bspline-g"
    "bspline-g-check"
    "bspline-h"
    "bspline-h-check"
#    "drr"
#    "drr-stats"
#    "drr-check"
    "fdk-cpu"
    "fdk-cpu-stats"
    "fdk-cpu-check"
    "fdk-cuda"
    "fdk-cuda-stats"
    "fdk-cuda-check"
    "plm-bspline-single-c"
    "plm-bspline-single-c-stats" 
    "plm-bspline-single-c-check" 
    "plm-bspline-single-f"
    "plm-bspline-single-f-stats" 
    "plm-bspline-single-f-check" 
    "plm-bspline-openmp"
    "plm-bspline-openmp-stats" 
    "plm-bspline-openmp-check" 
    "plm-bspline-cuda"
    "plm-bspline-cuda-stats" 
    "plm-bspline-cuda-check" 
    "plm-convert-dicom"
    "plm-convert-dicom-stats"
    "plm-convert-dicom-check"
    "plm-convert-dicom-rtss-a"
    "plm-convert-dicom-rtss-stats-a"
    "plm-convert-dicom-rtss-check-a"
    "plm-convert-dicom-rtss-b"
    "plm-convert-dicom-rtss-stats-b"
    "plm-convert-dicom-rtss-check-b"
    "plm-itk-translation"
    "plm-itk-translation-stats"
    "plm-itk-translation-check"
    "plm-resample-a"
    "plm-warp-a"
    "plm-warp-a-stats"
    "plm-warp-a-check"
    "plm-warp-b"
    "plm-warp-b-stats"
    "plm-warp-b-check"
    "plm-warp-c"
    "plm-warp-c-stats"
    "plm-warp-c-check"
    "tps-warp"
    "xf-to-xf"
    )
ENDIF (REDUCED_TEST)
