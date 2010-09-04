## Some good info on using CTest:
##   http://www.cmake.org/pipermail/cmake/2007-April/013797.html
## Apparently this is how to make the CMake script return status
##   http://www.cmake.org/Wiki/CMake/C_Plugins_for_Loadable_Commands

set(ENV{PATH} "$ENV{PATH};${ITK_LIBRARY_PATH};${PLM_PLASTIMATCH_PATH};c:/Program Files/Microsoft DirectX SDK (June 2007)/Utilities/Bin/x86")

# Only used for brook programs
set(ENV{BRT_RUNTIME} "dx9")

MESSAGE("PLM_TEST_COMMAND is ${PLM_TEST_COMMAND}")
MESSAGE("PARMS is ${PARMS}")

IF(WORKING_DIR)
  SET(WORKING_DIR WORKING_DIRECTORY ${WORKING_DIR})
ENDIF(WORKING_DIR)

EXECUTE_PROCESS(
  COMMAND ${PLM_TEST_COMMAND} ${PARMS}
  ${WORKING_DIR}
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE STDOUT
  ERROR_VARIABLE STDERR
)

MESSAGE("RETVAL: ${RESULT}")
MESSAGE("STDOUT: ${STDOUT}")
MESSAGE("STDERR: ${STDERR}")

SET(CMD_FN "${PLM_TESTING_BUILD_DIR}/${TESTNAME}.cmd")
FILE(WRITE ${CMD_FN} ${PLM_TEST_COMMAND})
FOREACH(PARM ${PARMS})
  FILE(APPEND ${CMD_FN} " \"" ${PARM} "\"")
ENDFOREACH(PARM ${PARMS})
FILE(APPEND ${CMD_FN} "\n")

SET(STDOUT_FN "${PLM_TESTING_BUILD_DIR}/${TESTNAME}.stdout.txt")
FILE(WRITE ${STDOUT_FN} ${STDOUT})

SET(STDERR_FN "${PLM_TESTING_BUILD_DIR}/${TESTNAME}.stderr.txt")
FILE(WRITE ${STDERR_FN} ${STDERR})

IF(${RESULT} EQUAL 0)
  MESSAGE("Not an error")
ELSE(${RESULT} EQUAL 0)
  MESSAGE(SEND_ERROR "An error")
ENDIF(${RESULT} EQUAL 0)
