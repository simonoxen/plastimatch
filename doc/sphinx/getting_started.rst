Getting started
====================

Downloading the code
--------------------

The recommended method for downloading plastimatch is to use subversion
to download the source, and compile from source.  
If you aren't yet familiar with subversion, you can read about it on the 
subversion web site:

  http://subversion.tigris.org/

To download using subversion, use the following command::

  $ svn co http://forge.abcd.harvard.edu/svn/plastimatch/plastimatch/trunk plastimatch

You will need to supply the user name and password::

  User: anonymous
  Password: <empty>

If you have already downloaded a previous version, 
you can update to the latest version by executing the following command 
from within the plastimatch source directory::

  $ svn update

Build dependencies
------------------

Cmake (required)
^^^^^^^^^^^^^^^^
Plastimatch uses cmake, so you must download and install cmake 
before you can build plastimatch.  Download from here:

  http://cmake.org/

Cmake 2.6 or higher is required.  Cmake 2.8 is required if you 
want to build the Slicer plugin.

Compiler (required)
^^^^^^^^^^^^^^^^^^^
You will need a C/C++ compiler.  If you are running 
Windows, we recommend Microsoft Visual Studio (Express or Full, 
2005 or 2008).  If you are running unix, we recommend gcc.
You can download the Microsoft Visual Studio Express compiler 
from here:

  http://www.microsoft.com/Express/

For the Express version, you need the platform SDK as well.

ITK (required)
^^^^^^^^^^^^^^
ITK is required for the main plastimatch program.  But if you only 
want the DRR and FDK programs, you don't need it.  Get ITK from here:

  http://itk.org/

ITK 3.16.0 (or higher) is recommended.

When you build ITK, the following settings are recommended::

  CMAKE_BUILD_TYPE                          Release
  BUILD_EXAMPLES                            OFF
  BUILD_SHARED_LIBS                         (EITHER)
  BUILD_TESTING                             OFF
  ITK_USE_REVIEW                            ON
  ITK_USE_OPTIMIZED_REGITRATION_METHODS     ON

Fortran or f2c (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Fortran (or f2c) is required if you want to run B-Spline registration with 
LBFGSB optimization.  
For windows, we recommend the f2c library, which is already 
included in plastimatch.
On unix, we recommend the GNU Fortran compiler, which will be found 
automatically if it is installed.
The f2c library may be used if no fortran compiler is available.

  http://www.netlib.org/f2c/

Cuda (recommended)
^^^^^^^^^^^^^^^^^^
Cuda is needed if you want GPU acceleration for FDK and B-Spline 
registration.  Install all three components: driver, toolkit, and SDK.
The following versions of CUDA are known to work::

  CUDA 2.1              Status unknown
  CUDA 2.2              Works
  CUDA 2.3              Works

Download CUDA from here:

  http://www.nvidia.com/object/cuda_get.html

DCMTK (optional)
^^^^^^^^^^^^^^^^
DCMTK is needed for mondoshot and a few other small utilities.  On Unix, 
it is a breeze, but Windows can be tricky.  Here is a rough guide how 
to compile and use on windows:

#. Download 3.5.4
#. Optionally, edit the CMakeLists.txt file distributed by dcmtk.  
   Delete (or comment out) the sections beginning with "settings for 
   Microsoft Visual C", and "settings for Microsoft Visual C++"
   (I think this is required when building shared library version 
   on MSVC 2005, but not required for static library on MSVC 2008).
#. Build
#. Install
#. Tell plastimatch to use the install directory

I once was able to use the contributed md-libraries (with VC 2005), 
but can no longer figure out how to do this.

Brook (optional)
^^^^^^^^^^^^^^^^
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

Slicer (optional)
^^^^^^^^^^^^^^^^^
Slicer is needed if you want to build the Slicer plugin.  
Download Slicer from here:

  http://slicer.org/

When building with Slicer, you should link with Slicer's ITK 
rather than linking with a separate one.  Leave the ITK directory blank 
when you configure.

WxWidgets (optional)
^^^^^^^^^^^^^^^^^^^^
WxWidgets is needed if you want to build "Mondoshot", the dicom screen 
capture program.  Download WxWidgets from here:

  http://wxwidgets.org


Compiling plastimatch (Windows)
-------------------------------
Run CMake as follows:

#. Select source directory and binary directory
#. Click configure
#. Select makefile format (e.g. MS VC 2005)
#. <CMake configures>
#. Set the ITK directory (sometimes it might be found automatically)
#. Set directories for optional components (such as slicer)
#. Click configure
#. <CMake configures>
#. Click OK
#. <CMake generates>

Then build in Visual Studio as follows:

#. Navigate to your binary directory
#. Open the project file plastimatch.sln into MSVC.  
#. Change the build type (e.g. release, debug) to match ITK (and other 
   dependencies.  You probably want release.
#. Click "Build Solution".  Let the project build.

Special instructions for building with brook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If using brook, make sure the plastimatch source directory doesn't 
have any spaces in the path.  Compiling the .br files with 
brook/fxc has problems with these.

Also, you might need to click "Build Solution" a twice in Visual Studio.
This is needed because brcc returns an error code for ARB & PS20 targets, 
which stops the build, even though the .cpp is generated correctly.
The second time around, the .cpp files exist and are up-to-date, 
and the build should continue correctly.

Special instructions for running cmake with MSYS/gcc on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There is a trick to building with MSYS/gcc.  
The trick is that you need to run the win32 cmake from 
the MSYS command line instead of the GUI.  For example, here is 
the command that I use::

   $ mkdir /c/gcs6/build/mingw/plastimatch
   $ cd /c/gcs6/build/mingw/plastimatch
   $ /c/Program\ Files/CMake\ 2.6/bin/cmake \
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

Compiling plastimatch (Unix)
----------------------------

Run CMake as follows:

#. mkdir /path/to/build/files; cd /path/to/build/files
#. ccmake /path/to/source/files
#. Type "c" to configure
#. <CMake configures>
#. Set the ITK directory (sometimes it might be found automatically)
#. Set directories for optional components (such as slicer)
#. You probably want to change the build type to "Release" (type it in)
#. Type "c" to configure
#. <CMake configures>
#. Type "g" to generate
#. <CMake generates>

Then build as follows:

#. Navigate to the plastimatch binary directory
#. Type "make"

