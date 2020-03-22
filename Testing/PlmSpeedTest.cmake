## This script runs the speed tests

if (NOT EXISTS "${PLM_BUILD_TESTING_DIR}/speedtest-1.mha")
  execute_process (
    COMMAND "${PLM_PLASTIMATCH_PATH}/plastimatch" "synth" "--output" "${PLM_BUILD_TESTING_DIR}/speedtest-1.mha" "--output-type" "float" "--pattern" "gauss" "--dim" "400 400 400" "--background" "-1000" "--foreground" "0"
    OUTPUT_VARIABLE STDOUT
    ERROR_VARIABLE STDERR
    )
  execute_process (
    COMMAND "${PLM_PLASTIMATCH_PATH}/plastimatch" "synth" "--output" "${PLM_BUILD_TESTING_DIR}/speedtest-2.mha" "--output-type" "float" "--pattern" "gauss" "--dim" "400 400 400" "--background" "-1000" "--foreground" "0" "--gauss-std" "50 30 10"
    OUTPUT_VARIABLE STDOUT
    ERROR_VARIABLE STDERR
    )
endif ()

execute_process (
  COMMAND "${PLM_PLASTIMATCH_PATH}/plastimatch" "register" "${PLM_BUILD_TESTING_DIR}/speedtest-a.txt"
  OUTPUT_VARIABLE STDOUT
  ERROR_VARIABLE STDERR
  )
string (REGEX MATCH "\\[ *([0-9.]*) s" JUNK "${STDOUT}")
message (STATUS "Time = ${CMAKE_MATCH_1}")
