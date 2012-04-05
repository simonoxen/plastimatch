SET (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/work/plastimatch")
SET (CTEST_BINARY_DIRECTORY "$ENV{HOME}/build/nightly/gelato_4/plastimatch")
SET (CTEST_CMAKE_COMMAND "/usr/local/bin/cmake")
SET (CTEST_COMMAND "/usr/local/bin/ctest -D Nightly")
SET (CTEST_INITIAL_CACHE "
//Name of generator.
CMAKE_GENERATOR:INTERNAL=Unix Makefiles

//Name of the build
BUILDNAME:STRING=04-lin64-PisR-Cd-gcc4.4.5

//Name of the computer/site where compile is being run
SITE:STRING=gelato

//The directory containing ITKConfig.cmake.  This is either the
// root of the build tree, or PREFIX/lib/InsightToolkit for an
// installation.
ITK_DIR:PATH=/home/gcs6/build/itk-3.20.1

//The directory containing a CMake configuration file for VTK.
VTK_DIR:PATH=/PHShome/gcs6/build/vtk-5.6.1

//Disable reg23 testing (it builds, but doesn't run on this computer)
ORAIFUTILS_BUILD_TESTING:BOOL=OFF
REG23_BUILD_TESTING:BOOL=OFF

//Build with shared libraries.
BUILD_SHARED_LIBS:BOOL=OFF
")
