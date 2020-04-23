## From Jan Woetzel, CMAKE email list
## http://www.cmake.org/pipermail/cmake/2006-January/007883.html
IF (UNIX)
  ADD_CUSTOM_TARGET(etags etags --members --declarations  `find ${CMAKE_SOURCE_DIR} -name *.cc -or -name *.cxx -or -name *.hxx -or -name *.hh -or -name *.cpp -or -name *.h -or -name *.c -or -name *.f`)
  ADD_CUSTOM_TARGET(gtags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E env "GTAGSFORCECPP=1" gtags)
  MESSAGE (STATUS "Etags targets added.")
ENDIF (UNIX)
