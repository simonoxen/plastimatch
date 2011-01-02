PROJECT (getopt)

SET (GETOPT_SRC
  getopt_long.c
  getopt.h
  )

ADD_LIBRARY (getopt STATIC ${GETOPT_LIBRARY_SRC})
