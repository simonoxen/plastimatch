## Some good info on using CTest:
##   http://www.cmake.org/pipermail/cmake/2007-April/013797.html
## Apparently this is how to make the CMake script return status
##   http://www.cmake.org/Wiki/CMake/C_Plugins_for_Loadable_Commands

set(ENV{PATH} "$ENV{PATH};${ITK_LIBRARY_PATH};c:/Program Files/Microsoft DirectX SDK (June 2007)/Utilities/Bin/x86")

#MESSAGE("PATH is $ENV{PATH}")
MESSAGE("PLM_TEST_COMMAND is ${PLM_TEST_COMMAND}")

#FILE(WRITE deleteme.txt "Hello world")

IF(WORKING_DIR)
  SET(WORKING_DIR WORKING_DIRECTORY ${WORKING_DIR})
ENDIF(WORKING_DIR)

EXECUTE_PROCESS(
  #COMMAND c:/gcs6/build/plastimatch-3.8.0/release/plastimatch
  COMMAND ${PLM_TEST_COMMAND} ${PARMS}
  COMMAND ${PLM_TEST_COMMAND} 
  ${WORKING_DIR}
  #WORKING_DIRECTORY C:/gcs6/idata/synth_mse
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE STDOUT
  ERROR_VARIABLE STDERR
)

MESSAGE("RETVAL: ${RESULT}")
MESSAGE("STDOUT: ${STDOUT}")
MESSAGE("STDERR: ${STDERR}")

IF(${RESULT} EQUAL 0)
  MESSAGE("Not an error")
ELSE(${RESULT})
  MESSAGE(SEND_ERROR "An error")
ENDIF(${RESULT})
