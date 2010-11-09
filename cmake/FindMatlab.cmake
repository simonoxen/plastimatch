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

