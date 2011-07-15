SET (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/work/plastimatch")
SET (CTEST_BINARY_DIRECTORY "$ENV{HOME}/build/nightly/plastimatch")
SET (CTEST_CMAKE_COMMAND "c:/Program Files/CMake 2.8/bin/cmake.exe")
SET (CTEST_COMMAND "c:/Program Files/CMake 2.8/bin/ctest.exe -D Nightly")
SET (CTEST_INITIAL_CACHE "
//Name of generator.
CMAKE_GENERATOR:INTERNAL=Visual Studio 9 2008

//Name of the build
BUILDNAME:STRING=win32-vs9

//Name of the computer/site where compile is being run
SITE:STRING=W0109966

//Build with shared libraries.
BUILD_SHARED_LIBS:BOOL=ON

//Path to the CVS
CVSCOMMAND:FILEPATH=C:/cygwin/bin/cvs.exe
SVNCOMMAND:FILEPATH=C:/cygwin/bin/svn.exe
")
