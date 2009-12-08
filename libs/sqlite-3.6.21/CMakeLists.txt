PROJECT (sqlite3)

SET (SQLITE_LIBRARY_SRC
  sqlite3.c
  sqlite3.h
  sqlite3ext.h
  )

ADD_LIBRARY (sqlite3 STATIC ${SQLITE_LIBRARY_SRC})
INSTALL (TARGETS sqlite3 DESTINATION lib)
