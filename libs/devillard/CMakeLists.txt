PROJECT (devillard)

# Lists of sources for executables
SET (LIBDEVILLARD_SRC
#  quickselect.c
  wirth.cpp
  )

# Build
add_library (devillard STATIC ${LIBDEVILLARD_SRC})
