Getting started
===============

Getting help
------------

Don't panic!

If you have problems downloading or installing plastimatch, 
please send an email to our mailing list.  We're friendly people.

  http://groups.google.com/group/plastimatch

Downloading the code
--------------------

The recommended method for downloading plastimatch is to use subversion
to download the source code, and then compile the source.
To download using subversion, use the following command::

  $ svn co http://forge.abcd.harvard.edu/svn/plastimatch/plastimatch/trunk plastimatch

You will need to supply the user name and password::

  User: anonymous
  Password: <empty>

If you are using Windows, you will need a subversion client.  
We recommend cygwin (http://cygwin.com) for command-line users, 
and TortoiseSVN (http://tortoisesvn.tigris.org) for graphical users.

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

C/C++ Compiler (required)
^^^^^^^^^^^^^^^^^^^^^^^^^
You will need a C/C++ compiler.  If you are running 
Windows, we recommend Microsoft Visual Studio (Express or Full, 
2005 or 2008).  If you are running unix, we recommend gcc.
You can download the Microsoft Visual Studio Express compiler 
from here:

  http://www.microsoft.com/Express/

Microsoft Visual Studio 2010 is also fine, but you will not 
be able to use CUDA.  

ITK (required)
^^^^^^^^^^^^^^
ITK is required for the main plastimatch program.  But if you only 
want the DRR and FDK programs, you don't need it.  Get ITK from here:

  http://itk.org/

Be careful of versions when using ITK.  We recommend the 
following versions:

+--------------------+-----------------------------+---------------------------+
|ITK Version         |Windows                      |Linux                      |
+====================+=============================+===========================+
|3.14.0 and earlier  |Not recommended              |Not recommended            |
+--------------------+-----------------------------+---------------------------+
|3.16.0              |Recommended                  |Recommended                |
+--------------------+-----------------------------+---------------------------+
|3.18.0              |Not recommended              |Recommended                |
+--------------------+-----------------------------+---------------------------+
|3.20.0              |Recommended                  |Recommended                |
+--------------------+-----------------------------+---------------------------+

When you build ITK, the following settings are recommended::

  CMAKE_BUILD_TYPE                          Release
  BUILD_EXAMPLES                            OFF
  BUILD_SHARED_LIBS                         (EITHER)
  BUILD_TESTING                             OFF
  ITK_USE_REVIEW                            ON
  ITK_USE_OPTIMIZED_REGITRATION_METHODS     ON

CUDA (recommended)
^^^^^^^^^^^^^^^^^^
CUDA is needed if you want GPU acceleration of the DRR, FDK, and B-Spline 
registration codes.  
You need to install the driver and toolkit, but the SDK is not needed.

Please note that CUDA is constantly evolving in order to provide new
high performance computing features. Because Plastimatch tends to utilize new
features as they become available, your CUDA drivers and toolkit need to be
relatively current.  The following table will help you with selecting the
correct CUDA version to install/upgrade::

  CUDA 2.1              Not supported
  CUDA 2.2              Not supported
  CUDA 2.3              Not supported
  CUDA 3.0              Recommended
  CUDA 3.1              Recommended

Download CUDA from here:

  http://developer.nvidia.com/object/cuda_archive.html

3D Slicer (optional)
^^^^^^^^^^^^^^^^^^^^
3D Slicer is needed if you want to build the Slicer plugin.  
Download Slicer from here:

  http://slicer.org/

See the section below for detailed instructions on how to build the 
3D Slicer plugin.

FFTW (optional)
^^^^^^^^^^^^^^^
The FFTW library is used to implement the ramp filter for FDK 
cone-beam reconstruction.  So if you are not using the FDK code, 
you don't need this.  We recommend the most current version of FFTW 3.

  http://www.fftw.org/

On windows, the precompiled DLLs work fine.  
However, you do need to create the import libraries.  
See this page for details:

  http://www.fftw.org/install/windows.html  

Fortran (optional)
^^^^^^^^^^^^^^^^^^
Plastimatch requires fortran, which can be satisfied with either 
a real fortran compiler, or with the f2c library.  If neither of these 
are installed, plastimatch supplies its own version of f2c.  You can 
hint which of these is used using the following CMake options::

  Option                 Default      Description
  ------                 -------      ------------
  PLM_PREFER_F2C         OFF          Prefer the f2c library over fortran
  PLM_PREFER_SYSTEM_F2C  ON           Prefer the system f2c library over
                                        the included f2c library

DCMTK (optional)
^^^^^^^^^^^^^^^^
DCMTK is needed for mondoshot and a few other small utilities.  On Unix, 
it is a breeze, but Windows can be tricky.  My experience is 
that the pre-built binaries don't seem to work, and you will 
get the best results if you build it yourself.
Here is a rough guide how 
to compile and use on windows:

#. Download and unpack source code for 3.5.4
#. Run CMake - set WITH_LIBPNG, WITH_LIBTIFF, WITH_ZLIB to OFF
#. Build
#. Install - this will create a directory "dcmtk-3.5.4-win32-i386" 
   with the same parent as the source directory
#. Run CMake on plastimatch - set DCMTK_DIR to the install directory

I once was able to use the contributed md-libraries (with VC 2005), 
but can no longer figure out how to do this.

WxWidgets (optional)
^^^^^^^^^^^^^^^^^^^^
WxWidgets is needed if you want to build "Mondoshot", the dicom screen 
capture program.  Download WxWidgets from here:

  http://wxwidgets.org

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

   $ mkdir /c/gcs6/build/plastimatch-mingw
   $ cd /c/gcs6/build/plastimatch-mingw
   $ /c/Program\ Files/CMake\ 2.8/bin/cmake \
       -DITK_DIR=/c/gcs6/build/itk-mingw \
       -G"MSYS Makefiles" \
       /c/gcs6/projects/plastimatch

Then, edit CMakeCache.txt to set your options.  Re-run cmake 
to create the MSYS Makefile, and then run make to build.

Special instructions for Visual Studio 2010
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The CUDA compiler nvcc is not compatible with Visual Studio 2010.
That is why we use Visual Studo 2008.  But, if you 
insist on using VS 2010, there are some workarounds
(Google is your friend).

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

   Users with multicore systems can speed up the process of compiling
   plastimatch considerably by invoking make with the -j option.  For
   example, a user with a dual-core system would type:

   make -j 2

   whereas a user with a quad-core system would type:

   make -j 4


Compiling the 3D Slicer extensions
----------------------------------
METHOD ONE:

#. Build slicer from source.  Use slicer 3.6, not slicer 4.

   http://www.slicer.org/slicerWiki/index.php/Slicer3:Build_Instructions

   If you are on Vista, you need to turn off UAC.
   If you are on Vista or 7, you need to run cygwin as administrator
   I suggest these options::

     ./Slicer3/Scripts/getbuildtest.tcl --release -t ""

   The slicer build takes a while.  Let it run over night.

#. Run slicer, just make sure the build went ok.

#. Make a new build directory for plastimatch.  

#. Run CMake

   Configure.
   Set Slicer3_DIR to the Slicer3-build directory.
   You don't need to set ITK -- the script should use Slicer's ITK.
   Configure again.
   Generate.

#. Build plastimatch.  You should find the plugins here:

   lib/Slicer3/Plugins/Release

#. Fire up slicer.  You need to tell slicer where the plugins are located

   View -> Application Settings -> Module Settings
   Click on the "Add a preset" icon
   Browse to the lib/Slicer3/Plugins/Release directory
   Click Close
   Restart slicer

#. You should see the plastimatch plugin in the module selector

METHOD TWO:

#. Build 3D Slicer as described above.

#. Use slicer's extension builder script to make the plugin::

     ./Slicer3/Scripts/extend.tcl --release -t "" plastimatch-slicer

#. You should find the plugins here:

   Slicer3-ext/plastimatch-slicer-build/lib/Slicer3/Plugins/Release

#. Plugins get uploaded here:

   http://ext.slicer.org/ext/trunk

   Your plugin gets put in one of the subdirectories, organized by 
   the platform and the svn version number of slicer.  

#. Add module path as described above -OR- download using extension manager


.. JAS 09.03.2010
.. The below has been commented out because it is now automatically
.. performed by my PLM_nvcc-check.cmake script.

.. Special Instructions For Linux Systems Using gcc-4.4
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. These instructions are for Linux users who desire GPU acceleration via CUDA.
   Due to an incompatibility between the Nvidia CUDA Compiler (nvcc) and version
   4.4 of the GNU C Compiler (gcc), Linux users must ensure that gcc-4.3 is
   available and that nvcc is set to use it.  If your system already uses version
   4.3 of gcc by default (run gcc --version to check), you may ignore these
   instructions.

.. Debian/Ubuntu users may install gcc version 4.3 by running the following from
   the command console:

..  $ sudo apt-get install gcc-4.3

.. Now, within the CMake curses frontend (ccmake) hit 't' to toggle advanced mode
   ON.  You will be presented with many new flags.  Scroll down using the arrow
   keys until you find CUDA_NVCC_FLAGS.  Once CUDA_NVCC_FLAGS is selected, hit
   enter and type the following into the field:

..  --compiler-bindir=PATH_TO_GCC_4.3

.. For example, under Ubuntu 9.04 with gcc-4.3 installed, one would enter:

..  --compiler-bindir=/usr/bin/gcc-4.3

.. You can now hit 't' again to hide the advanced mode flags.  Now you can
   continue the build process as usual by pressing "c" to configure.
