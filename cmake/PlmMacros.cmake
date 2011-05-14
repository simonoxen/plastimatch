##-----------------------------------------------------------------------------
##  Macros for creating targets
##-----------------------------------------------------------------------------
## JAS 2011.01.24
## I have commented out the INSTALL for the PLM_ADD_LIBRARY
## macro since it was only serving to include static link
## libraries in the CPack generated packages.
## Namely: libplastimatch1.a, libgpuit.a, & libf2c_helper.a
## GCS 2011-04-19
## However, it is also needed to correctly install dlls for windows
## binary packaging.
macro (PLM_ADD_LIBRARY 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  if (BUILD_AGAINST_SLICER3)
    add_library (${TARGET_NAME} STATIC ${TARGET_SRC})
    slicer3_set_plugins_output_path (${TARGET_NAME})
  else ()
    add_library (${TARGET_NAME} ${TARGET_SRC})
    set_target_properties (${TARGET_NAME} PROPERTIES 
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    if (WIN32 AND BUILD_SHARED_LIBS)
      install (TARGETS ${TARGET_NAME} RUNTIME DESTINATION bin)
    endif ()
    if (PLM_INSTALL_LIBRARIES)
      install (TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
    endif ()
  endif ()
  target_link_libraries (${TARGET_NAME} ${TARGET_LIBS})
  if (NOT ${TARGET_LDFLAGS} STREQUAL "")
    set_target_properties(${TARGET_NAME} 
      PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
  endif ()
endmacro ()
    
macro (PLM_ADD_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS TARGET_INSTALL)

  IF (NOT BUILD_AGAINST_SLICER3)
    ADD_EXECUTABLE(${TARGET_NAME} ${TARGET_SRC})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} ${TARGET_LIBS})
    SET_TARGET_PROPERTIES(${TARGET_NAME} 
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    IF (NOT ${TARGET_LDFLAGS} STREQUAL "")
      SET_TARGET_PROPERTIES(${TARGET_NAME} 
	PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
    ENDIF ()
    # CXX linkage required for nlopt
    SET_TARGET_PROPERTIES (${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    IF (${TARGET_INSTALL})
      INSTALL(TARGETS ${TARGET_NAME} DESTINATION bin)
    ENDIF ()
  ENDIF ()
endmacro ()

macro (PLM_ADD_SLICER_EXECUTABLE 
    TARGET_NAME TARGET_SRC TARGET_LIBS TARGET_LDFLAGS)

  generateclp (${TARGET_SRC} ${TARGET_NAME}.xml)
  add_executable (${TARGET_NAME} ${TARGET_SRC})
  target_link_libraries (${TARGET_NAME} ${TARGET_LIBS})
  if (NOT ${TARGET_LDFLAGS} STREQUAL "")
    set_target_properties (${TARGET_NAME} 
      PROPERTIES LINK_FLAGS ${TARGET_LDFLAGS})
  endif ()
  slicer3_set_plugins_output_path (${TARGET_NAME})
  slicer3_install_plugins (${TARGET_NAME})
endmacro ()

macro (PLM_ADD_SLICER_MODULE 
    TARGET_NAME TARGET_SRC TARGET_LIBS)

  #GENERATELM (TARGET_SRC ${TARGET_NAME}.xml)
  add_library (${TARGET_NAME} ${TARGET_SRC})
  target_link_libraries (${TARGET_NAME} 
    ${Slicer_Libs_LIBRARIES}
    ${Slicer_Base_LIBRARIES}
    ${KWWidgets_LIBRARIES}
    ${ITK_LIBRARIES}
    ${TARGET_LIBS})

  if (SLICER_IS_SLICER3)
    slicer3_set_modules_output_path (${TARGET_NAME})
  endif ()
  #SLICER3_INSTALL_PLUGINS (${TARGET_NAME})
endmacro ()

macro (PLM_ADD_OPENCL_FILE SRCS CL_FILE)
  # I don't yet know how to bundle the .cl file within the executable.
  # Therefore, copy the .cl into binary directory.
  set (${SRCS} ${${SRCS}} "${CMAKE_BINARY_DIR}/${CL_FILE}")
  add_custom_command (
    OUTPUT "${CMAKE_BINARY_DIR}/${CL_FILE}"
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_CURRENT_SOURCE_DIR}/${CL_FILE}" 
    "${CMAKE_BINARY_DIR}/${CL_FILE}" 
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${CL_FILE}")
  # Need in the testing directory too :(
  set (${SRCS} ${${SRCS}} "${PLM_TESTING_BUILD_DIR}/${CL_FILE}")
  add_custom_command (
    OUTPUT "${PLM_TESTING_BUILD_DIR}/${CL_FILE}"
    COMMAND ${CMAKE_COMMAND} "-E" "copy" 
    "${CMAKE_CURRENT_SOURCE_DIR}/${CL_FILE}" 
    "${PLM_TESTING_BUILD_DIR}/${CL_FILE}" 
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${CL_FILE}")
endmacro ()

macro (PLM_SLICER_COPY_DLL TARGET SRC DEST DEPENDENCY)
  add_custom_target (${TARGET} ALL DEPENDS "${DEST}")
  add_custom_command (
      OUTPUT "${DEST}"
      COMMAND
      ${CMAKE_COMMAND} "-E" "copy"
      "${SRC}"
      "${DEST}"
      DEPENDS ${DEPENDENCY}
      )
endmacro ()

macro (PLM_SET_SSE2_FLAGS)
  foreach (SRC ${ARGN})
    # JAS 08.19.2010 - Unfortunately, this doesn't work.
    #  SET_PROPERTY(
    #      SOURCE bspline.c
    #      APPEND PROPERTY COMPILE_DEFINITIONS ${SSE2_FLAGS}
    #      )
    # So, we ask CMake more forcefully to add additional compile flags
    get_source_file_property (OLD_FLAGS ${SRC} COMPILE_FLAGS)
    if (OLD_FLAGS MATCHES "NONE")
      set (OLD_FLAGS "")
    endif ()
    set_source_files_properties (
      ${SRC} PROPERTIES COMPILE_FLAGS "${OLD_FLAGS} -msse2")
  endforeach ()
endmacro ()
