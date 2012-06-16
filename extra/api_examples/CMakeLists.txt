project (plastimatch_api_example)
cmake_minimum_required (VERSION 2.8)

find_package (Plastimatch)

# If you have multiple ITK installed on the computer, use the one 
# which plastimatch is linked with.
set (ITK_DIR "${PLASTIMATCH_ITK_DIR}")
find_package (ITK)

if (PLASTIMATCH_FOUND)
  include (${ITK_USE_FILE})
  include_directories (AFTER ${PLASTIMATCH_INCLUDE_DIRS})
  add_executable (plastimatch_example main.cxx)
  target_link_libraries (plastimatch_example ${PLASTIMATCH_LIBRARIES})
else ()
  message (STATUS "Warning: plastimatch not found")
endif ()
