
SET (CUDA_FOUND @CUDA_FOUND@)
SET (BROOK_FOUND @BROOK_FOUND@)
SET (PLM_TEST_BSPLINE_FLAVORS @PLM_TEST_BSPLINE_FLAVORS@)
SET (PLM_TEST_DICOM @PLM_TEST_DICOM@)
SET (CMAKE_Fortran_COMPILER_WORKS @CMAKE_Fortran_COMPILER_WORKS@)
SET (PLM_TESTING_BUILD_DIR "@PLM_TESTING_BUILD_DIR@")
SET (PLM_PLASTIMATCH_PATH_HACK "@PLM_PLASTIMATCH_PATH_HACK@")

#SET (REDUCED_TEST ON)
SET (REDUCED_TEST OFF)

## If we don't have functioning GPU, don't run cuda tests
SET (RUN_CUDA_TESTS OFF)
IF (CUDA_FOUND)
  EXECUTE_PROCESS (COMMAND 
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

# drr-d, landmark-warp doen't work yet
SET (CTEST_CUSTOM_TESTS_IGNORE
  ${CTEST_CUSTOM_TESTS_IGNORE}
  "drr-d"
  "drr-d-stats"
  "drr-d-check"
  "landmark-warp"
  )

## If we don't have a fortran compiler, don't test bragg_curve
IF (NOT CMAKE_Fortran_COMPILER_WORKS)
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "bragg-curve"
    "bragg-curve-check"
    )
ENDIF (NOT CMAKE_Fortran_COMPILER_WORKS)

## If we didn't get dicom test data, don't run dicom tests
IF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "plm-convert-dicom-a"
    "plm-convert-dicom-a-stats"
    "plm-convert-dicom-a-check"
    "plm-convert-dicom-b"
    "plm-convert-dicom-b-stats"
    "plm-convert-dicom-b-check"
    "plm-convert-dicom-c"
    "plm-convert-dicom-c-stats"
    "plm-convert-dicom-c-check"
    "plm-convert-dicom-d"
    "plm-convert-dicom-d-stats"
    "plm-convert-dicom-d-check"
    "plm-convert-cxt"
    "plm-convert-cxt-stats"
    "plm-convert-cxt-check"
    )
ENDIF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")

## If we didn't get xio test data, don't run xio tests
IF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")
  SET (CTEST_CUSTOM_TESTS_IGNORE
    ${CTEST_CUSTOM_TESTS_IGNORE}
    "plm-convert-xio"
    "plm-convert-xio-stats"
    "plm-convert-xio-check"
    "plm-warp-e"
    "plm-warp-e-stats-1"
    "plm-warp-e-check-1"
    "plm-warp-e-stats-2"
    "plm-warp-e-check-2"
    "plm-warp-e-stats-3"
    "plm-warp-e-check-3"
    "plm-warp-e-stats-4"
    "plm-warp-e-check-4"
    "plm-warp-f"
    "plm-warp-f-stats-1"
    "plm-warp-f-check-1"
    )
ENDIF (NOT EXISTS "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")

## If we didn't compile with cuda, don't run these tests
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

## If we didn't compile with brook, don't run these tests
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
    "synth-1"
    "synth-2"
    "synth-3"
    "synth-4"
    "rect-1"
    "rect-2"
    "synth-7"
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
    "bspline-mi-c-1"
    "bspline-mi-c-1-check"
    "bspline-mi-c-2"
    "bspline-mi-c-2-check"
    "drr-a"
    "drr-a-stats"
    "drr-a-check"
    "drr-b"
    "drr-b-stats"
    "drr-b-check"
    "drr-c"
    "drr-c-stats"
    "drr-c-check"
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
    "plm-convert-dicom-a"
    "plm-convert-dicom-a-stats"
    "plm-convert-dicom-a-check"
    "plm-convert-dicom-b"
    "plm-convert-dicom-b-stats"
    "plm-convert-dicom-b-check"
    "plm-convert-dicom-c"
    "plm-convert-dicom-c-stats"
    "plm-convert-dicom-c-check"
    "plm-convert-dicom-d"
    "plm-convert-dicom-d-stats"
    "plm-convert-dicom-d-check"
    "plm-convert-cxt"
    "plm-convert-cxt-stats"
    "plm-convert-cxt-check"
    "plm-convert-xio"
    "plm-convert-xio-stats"
    "plm-convert-xio-check"
    "plm-usage" 
    "plm-itk-translation"
    "plm-itk-translation-stats"
    "plm-itk-translation-check"
    "plm-bspline-single-c" 
    "plm-bspline-single-c-stats"
    "plm-bspline-single-c-check"
    "plm-bspline-single-f" 
    "plm-bspline-single-f-stats"
    "plm-bspline-single-f-check"
    "plm-bspline-single-h" 
    "plm-bspline-single-h-stats"
    "plm-bspline-single-h-check"
    "plm-bspline-openmp" 
    "plm-bspline-openmp-stats"
    "plm-bspline-openmp-check"
    "plm-bspline-cuda" 
    "plm-bspline-cuda-stats"
    "plm-bspline-cuda-check"
    "plm-resample-a"
    "plm-warp-a"
    "plm-warp-a-stats"
    "plm-warp-a-check"
    "plm-warp-b"
    "plm-warp-b-stats-1"
    "plm-warp-b-check-1"
    "plm-warp-b-stats-2"
    "plm-warp-b-check-2"
    "plm-warp-c"
    "plm-warp-c-stats"
    "plm-warp-c-check"
    "plm-warp-d" 
    "plm-warp-d-stats"
    "plm-warp-d-check"
#    "plm-warp-e" 
    "plm-warp-e-stats-1"
    "plm-warp-e-check-1"
    "plm-warp-e-stats-2"
    "plm-warp-e-check-2"
    "plm-warp-e-stats-3"
    "plm-warp-e-check-3"
    "plm-warp-e-stats-4"
    "plm-warp-e-check-4"
    "plm-warp-e-check-5"
    "plm-warp-f" 
    "plm-warp-f-stats-1"
    "plm-warp-f-check-1"
    "bragg-curve"
    "bragg-curve-check"
    "proton-dose"
    "tps-warp"
    "xf-to-xf-1"
    "xf-to-xf-2"
    "xf-to-xf-3"
    )
ENDIF (REDUCED_TEST)
