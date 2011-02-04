PROJECT (gengetopt)

INCLUDE (TestForSSTREAM)
INCLUDE (TestForSTDNamespace)

IF (NOT CMAKE_NO_ANSI_STRING_STREAM)
  ADD_DEFINITIONS (-DHAVE_SSTREAM)
ENDIF ()
IF (NOT CMAKE_NO_STD_NAMESPACE)
  ADD_DEFINITIONS (-DHAVE_NAMESPACES)
ENDIF ()

ADD_DEFINITIONS (-DPACKAGE="gengetopt" -DVERSION="2.22.4")

SET (GENGETOPT_SRC
  argsdef.c  gengetopt.cc  getopt.c  parser.cc   yyerror.cc
  cmdline.c  getopt1.c     gm.cc     scanner.cc
  acceptedvalues.cpp
  fileutils.cpp
  ggos.cpp
  gm_utils.cpp
  )

ADD_EXECUTABLE (gengetopt ${GENGETOPT_SRC})
