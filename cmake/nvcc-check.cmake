# James Shackleford
# Date: 08.24.2010
# File: PLM_nvcc-check.cmake
#
# Currently, nvcc only works with gcc-4.3
#
# This script:
#   * Checks the version of default gcc
#   * If not 4.3, we look for gcc-4.3 on the system
#   * If found we tell nvcc to use it
#   * If not found, we kill CMake and tell user to install gcc-4.3
#   * Checks for NVCC 3.2
#   * If found, we add -fpermissive to NVCCFLAGS
# 
# NOTE: If nvcc tries to use gcc-4.4 (for example) the build simply
#       fails.  Ending things at CMake with a request for gcc-4.3
#       is the most graceful failure I could provide.
######################################################################
IF(CUDA_FOUND)
    IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
        IF(CMAKE_COMPILER_IS_GNUCC)

            # Get the gcc version number
            EXEC_PROGRAM(gcc ARGS "-dumpversion" OUTPUT_VARIABLE GCCVER)

            # Get gcc's major and minor revs
            STRING(REGEX REPLACE "([0-9]+).[0-9]+.[0-9]+" "\\1" GCCVER_MAJOR "${GCCVER}")
            STRING(REGEX REPLACE "[0-9]+.([0-9]+).[0-9]+" "\\1" GCCVER_MINOR "${GCCVER}")
            STRING(REGEX REPLACE "[0-9]+.[0-9]+.([0-9]+)" "\\1" GCCVER_PATCH "${GCCVER}")

            #            MESSAGE(STATUS "nvcc-check: GCC Version is ${GCCVER_MAJOR}.${GCCVER_MINOR}.${GCCVER_PATCH}")

            # CUDA 2.X IS UNSUPPORTED
            IF (CUDA_VERSION_MAJOR MATCHES "2")
                MESSAGE(FATAL_ERROR "nvcc-check: Plastimatch only supports CUDA 3.0+\n")
            ENDIF ()

            # CUDA 3.X NEEDS GCC-4.3
            IF (CUDA_VERSION_MAJOR MATCHES "3")
                IF(GCCVER_MAJOR MATCHES "4")
                    IF(GCCVER_MINOR MATCHES "3")
                        MESSAGE(STATUS "nvcc-check: Found gcc-${GCCVER_MAJOR}.${GCCVER_MINOR}... success.")
                    ELSE()
                        MESSAGE(STATUS "nvcc-check: Found gcc-${GCCVER_MAJOR}.${GCCVER_MINOR}... searching for gcc-4.3")
                        EXEC_PROGRAM(which ARGS "gcc-4.3" OUTPUT_VARIABLE GCC43 RETURN_VALUE GCC43_EXIST)
    
                        IF()
                            MESSAGE(STATUS "nvcc-check: Found gcc-4.3... telling nvcc to use it!")
                            MESSAGE(STATUS "nvcc-check: CUDA_NVCC_FLAGS set to \"${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC43}\"")
                            SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC43})
                        ELSE()
                            MESSAGE(FATAL_ERROR "nvcc-check: Please install gcc-4.3, it is needed by nvcc \(CUDA\).\nNote that gcc-4.3 can be installed side-by-side with your current version of gcc (${GCCVER_MAJOR}.${GCCVER_MINOR}.${GCCVER_PATCH}).\nYou need not replace your current version of gcc; just make gcc-4.3 available as well so that nvcc can use it.\nDebian/Ubuntu users with root privilages may simply enter the following at a terminal prompt:\n  sudo apt-get install gcc-4.3 g++-4.3\n")
                        ENDIF()
    
                    ENDIF()
                ENDIF()
            ENDIF ()

            # CUDA 4.X NEEDS GCC-4.4 or GCC-4.3
            IF (CUDA_VERSION_MAJOR MATCHES "4")
                IF(GCCVER_MAJOR MATCHES "4")
                    # GCC 4.4 is okay
                    IF(GCCVER_MINOR MATCHES "4")
                        MESSAGE(STATUS "nvcc-check: Found gcc-${GCCVER_MAJOR}.${GCCVER_MINOR}... success.")
                    ELSE()
                        # GCC 4.3 is okay
                        IF(GCCVER_MINOR MATCHES "4")
                            MESSAGE(STATUS "nvcc-check: Found gcc-${GCCVER_MAJOR}.${GCCVER_MINOR}... success.")
                        ELSE()
                            # search for GCC-4.4
                            MESSAGE(STATUS "nvcc-check: Found gcc-${GCCVER_MAJOR}.${GCCVER_MINOR}... searching for gcc-4.4")
                            EXEC_PROGRAM(which ARGS "gcc-4.4" OUTPUT_VARIABLE GCC44 RETURN_VALUE GCC44_EXIST)
    
                            IF(GCC44_EXIST EQUAL 0)
                                MESSAGE(STATUS "nvcc-check: Found gcc-4.4... telling nvcc to use it!")
                                MESSAGE(STATUS "nvcc-check: CUDA_NVCC_FLAGS set to \"${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC44}\"")
                                SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC44})
                            ELSE()

                                # search for GCC-4.3
                                MESSAGE(STATUS "nvcc-check: Found gcc-${GCCVER_MAJOR}.${GCCVER_MINOR}... searching for gcc-4.3")
                                EXEC_PROGRAM(which ARGS "gcc-4.3" OUTPUT_VARIABLE GCC43 RETURN_VALUE GCC43_EXIST)
    
                                IF(GCC43_EXIST EQUAL 0)
                                    MESSAGE(STATUS "nvcc-check: Found gcc-4.3... telling nvcc to use it!")
                                    MESSAGE(STATUS "nvcc-check: CUDA_NVCC_FLAGS set to \"${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC43}\"")
                                    SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir=${GCC43})
                                ELSE()
                                    # didn't find anything useful
                                    MESSAGE(FATAL_ERROR "nvcc-check: Please install gcc-4.3, it is needed by nvcc \(CUDA\).\nNote that gcc-4.3 can be installed side-by-side with your current version of gcc (${GCCVER_MAJOR}.${GCCVER_MINOR}.${GCCVER_PATCH}).\nYou need not replace your current version of gcc; just make gcc-4.3 available as well so that nvcc can use it.\nDebian/Ubuntu users with root privilages may simply enter the following at a terminal prompt:\n  sudo apt-get install gcc-4.3 g++-4.3\n")
                                ENDIF()
                            ENDIF()
                        ENDIF()
                    ENDIF()
                ENDIF()
            ENDIF ()

        ENDIF()
    ENDIF()



    IF(CMAKE_SYSTEM_NAME MATCHES "Linux" OR "APPLE")

        # For CUDA 3.2: surface_functions.h does some non-compliant things...
        #               so we tell g++ to ignore them when called via nvcc
        #               by passing the -fpermissive flag through the nvcc
        #               build trajectory.  Unfortunately, nvcc will also
        #               blindly pass this flag to gcc, even though it is not
        #               valid... resulting in TONS of warnings.  So, we go
        #               version checking again, this time nvcc...
        # Get the nvcc version number
        MESSAGE(STATUS "nvcc-check: NVCC Version is ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")

        IF(CUDA_VERSION_MAJOR MATCHES "3")
            IF(CUDA_VERSION_MINOR MATCHES "2")
                SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options='-fpermissive')
                MESSAGE(STATUS "nvcc-check: CUDA 3.2 exception: CUDA_NVCC_FLAGS set to \"${CUDA_NVCC_FLAGS}\"")
            ENDIF()
        ENDIF()
    ENDIF()

ENDIF(CUDA_FOUND)
