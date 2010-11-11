SET (MATLAB_FOUND 0)

FIND_PROGRAM (MATLAB_EXE
  matlab
  )
IF (MATLAB_EXE)

  MESSAGE (STATUS "Probing matlab capabilities")
  FILE (WRITE "${CMAKE_BINARY_DIR}/probe_matlab.c"
    "#include \"mex.h\"
    void
    mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
    {
    mxArray *v = mxCreateDoubleMatrix (1, 1, mxREAL);
    double *data = mxGetPr (v);
    *data = 1.23456789;
    plhs[0] = v;
    }")
  EXECUTE_PROCESS (COMMAND
    "${MATLAB_EXE}" -nosplash -nodisplay -r "mex -v probe_matlab.c;exit;"
    TIMEOUT 20
    RESULT_VARIABLE MATLAB_RESULT
    OUTPUT_VARIABLE MATLAB_STDOUT
    ERROR_VARIABLE MATLAB_STDERR
    )
  STRING (REGEX MATCH "LDEXTENSION *= *([^ \n]*)" JUNK ${MATLAB_STDOUT})
  SET (MATLAB_LDEXTENSION "${CMAKE_MATCH_1}")

  MESSAGE (STATUS "Mex extension = ${MATLAB_LDEXTENSION}")
  
  IF (MATLAB_LDEXTENSION)
    SET (MATLAB_FOUND 1)
  ENDIF (MATLAB_LDEXTENSION)

ENDIF (MATLAB_EXE)

## Macro for compiling mex files
MACRO (MEX_TARGET
    TARGET_NAME TARGET_SRC TARGET_LIBS)
  IF (MATLAB_FOUND)
    SET (MEX_COMPILE_CMD "compile_${TARGET_NAME}")
    SET (MEX_COMPILE_M "${CMAKE_BINARY_DIR}/${MEX_COMPILE_CMD}.m")
    SET (MEX_COMPILE_CMAKE "${CMAKE_BINARY_DIR}/${TARGET_NAME}.cmake")
    SET (MEX_COMPILE_SRC "${CMAKE_SOURCE_DIR}/${TARGET_SRC}")
    SET (MEX_COMPILE_TGT "${CMAKE_BINARY_DIR}/${TARGET_NAME}${MATLAB_LDEXTENSION}")
    FILE (WRITE "${MEX_COMPILE_M}"
      "mex")

    GET_DIRECTORY_PROPERTY (INCLUDE_DIRS INCLUDE_DIRECTORIES)
    FOREACH (DIR ${INCLUDE_DIRS})
      FILE (APPEND "${MEX_COMPILE_M}"
	" -I${DIR}")
    ENDFOREACH (DIR ${INCLUDE_DIRECTORIES})

    FILE (APPEND "${MEX_COMPILE_M}" " -L${CMAKE_BINARY_DIR}")
    FILE (APPEND "${MEX_COMPILE_M}" " -L${CMAKE_BINARY_DIR}/libs/libf2c")
    GET_DIRECTORY_PROPERTY (LINK_DIRS LINK_DIRECTORIES)
    FOREACH (DIR ${LINK_DIRS})
      FILE (APPEND "${MEX_COMPILE_M}"
	" -L${DIR}")
    ENDFOREACH (DIR ${LINK_DIRS})

    FOREACH (LIB ${TARGET_LIBS})
      FILE (APPEND "${MEX_COMPILE_M}"
	" -l${LIB}")
    ENDFOREACH (LIB ${TARGET_LIBS})

    FILE (APPEND "${MEX_COMPILE_M}"
      " \"${MEX_COMPILE_SRC}\";exit;\n")
    FILE (WRITE "${MEX_COMPILE_CMAKE}"
      "EXECUTE_PROCESS (COMMAND ${MATLAB_EXE} -nosplash -nodesktop -nojvm -nodisplay
       -r ${MEX_COMPILE_CMD}
       TIMEOUT 10
       RESULT_VARIABLE RESULT
       OUTPUT_VARIABLE STDOUT
       ERROR_VARIABLE STDERR)\n")
    ADD_CUSTOM_COMMAND (
      OUTPUT "${MEX_COMPILE_TGT}"
      COMMAND ${CMAKE_COMMAND} -P "${MEX_COMPILE_CMAKE}"
      DEPENDS "${MEX_COMPILE_SRC}")
    ADD_CUSTOM_TARGET (${TARGET_NAME}
      DEPENDS "${MEX_COMPILE_TGT}")
    #TARGET_LINK_LIBRARIES (${TARGET_NAME} ${MATLAB_LIBRARIES})
  ENDIF (MATLAB_FOUND)
ENDMACRO (MEX_TARGET)
