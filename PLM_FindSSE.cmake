# Some additional handeling should be added here so that the
# SSE2_FLAGS are set for the correct compiler.  Currently
# only works with gcc
SET (SSE2_FLAGS "-msse2")

# LINUX: We check for SSE extensions using a RegEx on /proc/cpuinfo
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")

    # Store /proc/cpuinfo output into CPUINFO
    EXEC_PROGRAM(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)

    # Check for SSE2
    STRING(REGEX REPLACE "^.*(sse2).*$" "\\1" SSE_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "sse2" "${SSE_THERE}" SSE2_TRUE)
    IF (SSE2_TRUE)
        SET(SSE2_FOUND true CACHE BOOL "SSE2 Available")
    ELSE (SSE2_TRUE)
        SET(SSE2_FOUND false CACHE BOOL "SSE2 Available")
    ENDIF (SSE2_TRUE)
    
    # Check for SSSE3
    STRING(REGEX REPLACE "^.*(ssse3).*$" "\\1" SSE_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "ssse3" "${SSE_THERE}" SSSE3_TRUE)
    IF (SSSE3_TRUE)
        SET(SSSE3_FOUND true CACHE BOOL "SSSE3 Available")
    ELSE (SSSE3_TRUE)
        SET(SSSE3_FOUND false CACHE BOOL "SSSE3 Available")
    ENDIF (SSSE3_TRUE)
    
    # Check for SSE4.1
    STRING(REGEX REPLACE "^.*(sse4_1).*$" "\\1" SSE_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "sse4_1" "${SSE_THERE}" SSE41_TRUE)
    IF (SSE41_TRUE)
        set(SSE4_1_FOUND true CACHE BOOL "SSE 4.1 Available")
    ELSE (SSE41_TRUE)
        set(SSE4_1_FOUND false CACHE BOOL "SSE 4.1 Available")
    ENDIF (SSE41_TRUE)

# OSX/DARWIN: We check for SSE extensions using a RegEx on sysctl output
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    EXEC_PROGRAM("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE CPUINFO)
    
    # Check for SSE2
    STRING(REGEX REPLACE "^.*(SSE2).*$" "\\1" SSE_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "SSE2" "${SSE_THERE}" SSE2_TRUE)
    IF (SSE2_TRUE)
        SET(SSE2_FOUND true CACHE BOOL "SSE2 Available")
    ELSE (SSE2_TRUE)
        SET(SSE2_FOUND false CACHE BOOL "SSE2 Available")
    ENDIF (SSE2_TRUE)

    # Check for SSSE3
    STRING(REGEX REPLACE "^.*(SSSE3).*$" "\\1" SSE_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "SSSE3" "${SSE_THERE}" SSSE3_TRUE)
    IF (SSSE3_TRUE)
        set(SSSE3_FOUND true CACHE BOOL "SSSE3 Available")
    ELSE (SSSE3_TRUE)
        set(SSSE3_FOUND false CACHE BOOL "SSSE3 Available")
    ENDIF (SSSE3_TRUE)
    
    # Check for SSE4.1
    STRING(REGEX REPLACE "^.*(SSE4.1).*$" "\\1" SSE_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "SSE4.1" "${SSE_THERE}" SSE41_TRUE)
    IF (SSE41_TRUE)
        set(SSE4_1_FOUND true CACHE BOOL "SSE4.1 Available ")
    ELSE (SSE41_TRUE)
        set(SSE4_1_FOUND false CACHE BOOL "SSE4.1 Available")
    ENDIF (SSE41_TRUE)

# WINDOWS: No autodetection.  User must manually enable SSE extensions
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(SSE2_FOUND false CACHE BOOL "SSE2 Available")
    set(SSSE3_FOUND  false CACHE BOOL "SSSE3 Available")
    set(SSE4_1_FOUND false CACHE BOOL "SSE4.1 Available")

# Something else... BSD Perhaps?
ELSE(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(SSE2_FOUND false CACHE BOOL "SSE2 Available")
    set(SSSE3_FOUND  false CACHE BOOL "SSSE3 Available")
    set(SSE4_1_FOUND false CACHE BOOL "SSE4.1 Available")
ENDIF(CMAKE_SYSTEM_NAME MATCHES "Linux")


IF (SSE2_FOUND)
    MESSAGE(STATUS "SSE2_FLAGS \"${SSE2_FLAGS}\"")
ELSE (SSE2_FOUND)
    MESSAGE(STATUS "CPU does not support SSE2.")
ENDIF(SSE2_FOUND)
    
#IF(NOT SSSE3_FOUND)
#    MESSAGE(STATUS "Could not find hardware support for SSSE3 on this machine.")
#ENDIF(NOT SSSE3_FOUND)

#IF(NOT SSE4_1_FOUND)
#    MESSAGE(STATUS "Could not find hardware support for SSE4.1 on this machine.")
#ENDIF(NOT SSE4_1_FOUND)

    
# Put these in the advanced toggles
mark_as_advanced(SSE2_FOUND SSSE3_FOUND SSE4_1_FOUND)
