SET (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/src/plastimatch-nightly")
SET (CTEST_BINARY_DIRECTORY "$ENV{HOME}/build/plastimatch-nightly")
SET (CTEST_CMAKE_COMMAND "/usr/bin/cmake -Wno-dev")
SET (CTEST_COMMAND "/usr/bin/ctest -D Nightly")
SET (CTEST_INITIAL_CACHE "
//Name of generator.
CMAKE_GENERATOR:INTERNAL=Unix Makefiles

//Name of the build
BUILDNAME:STRING=osx10.7.3-Pisr-Cd-gcc4.2-cuda4.1

//Name of the computer/site where compile is being run
SITE:STRING=tempest

//Build with shared libraries.
BUILD_SHARED_LIBS:BOOL=OFF

//PLM_CONFIG_DISABLE_REG23:BOOL=ON
")
