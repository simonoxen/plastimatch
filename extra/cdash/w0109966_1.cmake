SET (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/work/plastimatch")
SET (CTEST_BINARY_DIRECTORY "$ENV{HOME}/build/nightly/w0109966_1")
SET (CTEST_CMAKE_COMMAND "c:/Program Files/CMake 2.8/bin/cmake.exe")
SET (CTEST_COMMAND "c:/Program Files/CMake 2.8/bin/ctest.exe -D Nightly")
SET (CTEST_INITIAL_CACHE "
//Name of generator.
CMAKE_GENERATOR:INTERNAL=Visual Studio 9 2008

//Name of the build
BUILDNAME:STRING=w-01-wxp32-Pisr-Cd-vse9

//Name of the computer/site where compile is being run
SITE:STRING=w0109966

//Build with shared libraries.
BUILD_SHARED_LIBS:BOOL=ON

//The directory containing ITKConfig.cmake.  This is either the
// root of the build tree, or PREFIX/lib/InsightToolkit for an
// installation.
ITK_DIR:PATH=C:/gcs6/build/vs2008/itk-3.20.0

//Build REG-2-3
PLM_CONFIG_DISABLE_REG23:BOOL=ON

//Path to the CVS
CVSCOMMAND:FILEPATH=C:/cygwin/bin/cvs.exe
SVNCOMMAND:FILEPATH=C:/cygwin/bin/svn.exe
")
