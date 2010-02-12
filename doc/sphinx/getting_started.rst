Getting Started
====================

Downloading the code
--------------------

The recommended method for downloading plastimatch is to use subversion
to download the source, and compile from source.  
If you aren't yet familiar with subversion, you can read about it on the 
subversion web site:

http://subversion.tigris.org/

To download using subversion, use the following command:

  $ svn co http://forge.abcd.harvard.edu/svn/plastimatch/plastimatch/trunk plastimatch

You will need to supply the user name and password:

  User: anonymous
  Password: <empty>

If you have already downloaded a previous version, 
you can update to the latest version by executing the following command 
from within the plastimatch source directory:

  $ svn update

Build dependencies
------------------

Cmake
^^^^^
Plastimatch uses cmake, so you must download and install cmake 
before you can build plastimatch.  Download from here:

  http://cmake.org/

Cmake 2.6 or higher is required.  Cmake 2.8 is required if you 
want to build the Slicer plugin.

Compiler
^^^^^^^^
You will need a C/C++ compiler.  If you are running 
Windows, we recommend Microsoft Visual Studio (Express or Full, 
2005 or 2008).  If you are running unix, we recommend gcc.
You can download the Microsoft Visual Studio Express compiler 
from here:

  http://www.microsoft.com/Express/

For the Express version, you need the platform SDK as well.

ITK
^^^
ITK is required for the main plastimatch program.  But if you only 
want the DRR and FDK programs, you don't need it.  Get ITK from here:

  http://itk.org/

ITK 3.16.0 (or higher) is recommended.

When you build ITK, the following settings are recommended:

  BUILD_EXAMPLES                            OFF
  BUILD_SHARED_LIBS                         (EITHER)
  BUILD_TESTING                             OFF
  ITK_USE_REVIEW                            ON
  ITK_USE_OPTIMIZED_REGITRATION_METHODS     ON

Also, you probably want make sure the build type is "Release" when 
you compile.

Fortran (or f2c)
^^^^^^^^^^^^^^^^
Fortran is required if you want to run B-Spline registration with 
LBFGSB optimization.  Alternatively, f2c can be used.

For windows, the f2c library is included in plastimatch.
On unix, it is recommended to install the GNU Fortran compiler.  

If the f2c library is desired for other platforms (such as MSYS),
get from here:

  http://www.netlib.org/f2c/


DCMTK
^^^^^
DCMTK is needed for mondoshot and a few other small utilities.  On Unix, 
it is a breeze, but Windows can be tricky.  Here is a rough guide how 
to compile and use on windows:

  a) Download 3.5.4
  b) Optionally, edit the CMakeLists.txt file distributed by dcmtk.  
     Delete (or comment out) the sections beginning with "settings for 
     Microsoft Visual C", and "settings for Microsoft Visual C++"
     (I think this is required when building shared library version 
     on MSVC 2005, but not required for static library on MSVC 2008).
  c) Build
  d) Install
  e) Tell plastimatch to use the install directory

I once was able to use the contributed md-libraries (with VC 2005), 
but can no longer figure out how to do this.

Cuda
^^^^
Cuda is needed if you want GPU acceleration for FDK and B-Spline 
registration.  Install all three components: driver, toolkit, and SDK.

The following versions of CUDA are known to work:

  CUDA 2.1              Status unknown
  CUDA 2.2              Works
  CUDA 2.3              Believed to not work

Download CUDA from here:

  http://www.nvidia.com/object/cuda_get.html

Brook
^^^^^
Brook is depricated.

However, you still need brook for GPU acceleration of demons deformable 
registration.  Note also that brook is only tested on Windows.

Download Brook from here:

  http://graphics.stanford.edu/projects/brookgpu/

If using brook, you also need to install the DirectX SDK and 
the NVIDIA CG compiler.  Please refer to README.BROOK.TXT for details.

After you have compiled plastimatch, you need to set the brook runtime 
variables to get GPU acceleration.  Only the DirectX9 backend works.  
Using the %COMSPEC% shell, do this:

   set BRT_RUNTIME=dx9

Slicer
^^^^^^
Slicer is needed if you want to build the Slicer plugin.  This is still
experimental.  Download Slicer from here:

  http://slicer.org/

If you want the slicer plugin, you should link with Slicer's ITK 
rather than linking with a separate one.

Compiling plastimatch
---------------------

* If using brook, make sure the plastimatch source directory doesn't 
   have any spaces in the path (brook/fxc has problems with these).

* Windows instructions for running cmake

     Select source directory and binary directory
       Don't use any spaces in the path
     Click configure
     On windows, select makefile format (e.g. MS VC 2005)
       <CMake configures>
     Set ITK directory (might be found automatically)
     If using brook, set brook directory
     If using slicer, set Slicer3_DIR
     Click configure
       <CMake configures>
     Click OK
       <CMake generates>

* Unix instructions for running cmake

     mkdir /path/to/build/files; cd /path/to/build/files
     ccmake /path/to/source/files
     Type "c" to configure
       <CMake configures>
     Set ITK directory (might be found automatically)
     If using slicer, set Slicer3_DIR
     You probably want to change the build type to "Release" (type it in)
     Type "c" to configure
       <CMake configures>
     Type "g" to generate
       <CMake generates>

* Special instructions for running cmake with MSYS/gcc on Windows

   There is a trick to this: you need to run the win32 cmake from 
   the MSYS command line instead of the GUI.  For example, here is 
   the command that I use:

   mkdir /c/gcs6/build/mingw/plastimatch-3.14.0
   cd /c/gcs6/build/mingw/plastimatch-3.14.0
   /c/Program\ Files/CMake\ 2.6/bin/cmake \
      -DITK_DIR=/c/gcs6/build/mingw/itk-3.14.0 \
      -DF2C_LIBRARY=/c/gcs6/build/mingw/f2c/libf2c.a \
      -G"MSYS Makefiles" \
      /c/gcs6/projects/plastimatch

   Then, edit CMakeCache.txt to set your options.  Re-run cmake 
   to create the MSYS Makefile.

   Note, you can't use the included f2c binary libraries (vcf2c_libcmt.lib
   and vcf2c_msvcrt.lib).  You have to compile a separate version.

   Also, some versions of cmake seem to have a bug where they do not 
   correctly set the options for CMAKE_C_FLAGS_DEBUG, CMAKE_C_FLAGS_RELEASE, 
   and so on.  If this happens, you can copy these values from the CXX options.

* General windows instructions:

   1) Navigate to your binary directory

   2) Open the project file plastimatch.sln into MSVC.  

   3) Change the build type (e.g. release, debug) to match ITK and brook.  
      You probably want release.

   4) Click "Build Solution".  Let the project build.

   5) If using brook, you might need to click "Build Solution" a second time.  
      This is needed because brcc returns an error code for ARB & PS20 targets 
      which stops the build, even though the .cpp is generated correctly.

* General unix instructions:

   1) Navigate to the plastimatch binaries directory, and then type "make".

