PROJECT (liblbfgs)

SET (LIBLBFGS_LIBRARY_SRC
  lib/lbfgs.c
  lib/arithmetic_ansi.h
  lib/arithmetic_sse_float.h
  lib/arithmetic_sse_double.h
  )
INCLUDE_DIRECTORIES (BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ADD_LIBRARY (lbfgs STATIC ${LIBLBFGS_LIBRARY_SRC})
INSTALL (TARGETS lbfgs DESTINATION lib)
