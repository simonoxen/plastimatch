# - Find SQLite3
# Find the native Sqlite includes and library
#
#  SQLite3_INCLUDE_DIR - where to find zlib.h, etc.
#  SQLite3_LIBRARIES   - List of libraries when using zlib.
#  SQLite3_FOUND       - True if zlib found.


IF (SQLite3_INCLUDE_DIR)
  # Already in cache, be silent
  SET (Sqlite_FIND_QUIETLY TRUE)
ENDIF (SQLite3_INCLUDE_DIR)

FIND_PATH(SQLite3_INCLUDE_DIR sqlite3.h)

SET (SQLite3_NAMES sqlite3)
FIND_LIBRARY (SQLite3_LIBRARY NAMES ${SQLite3_NAMES})

# handle the QUIETLY and REQUIRED arguments and set SQLite3_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (SQLite3 DEFAULT_MSG 
  SQLite3_LIBRARY 
  SQLite3_INCLUDE_DIR)

IF(SQLite3_FOUND)
  SET (SQLite3_LIBRARIES ${SQLite3_LIBRARY})
ELSE (SQLite3_FOUND)
  SET (SQLite3_LIBRARIES)
ENDIF (SQLite3_FOUND)

MARK_AS_ADVANCED (SQLite3_LIBRARY SQLite3_INCLUDE_DIR)
