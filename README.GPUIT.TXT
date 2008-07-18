Thank you for downloading the GPU Imaging Toolkit.
See COPYRIGHT.TXT for copyright information.
See LICENSE.TXT for information on using and contributing.

GPUIT is developed and has been tested under Windows 2000 and 
Windows XP using MSVC 2005.

***** How to compile BROOK with MSVC 2005 *****
******* (Last tested Fri, Jun 18, 2008) *******

1)  Download and install Microsoft DirectX SDK and NVIDIA CG compiler

    The November 2007 version of the DirectX SDK should not be used, 
    since it has a bug in fxc.  The June 2007 version is recommded.

2)  Download brook:
      C:\gsharp\build>svn co https://brook.svn.sourceforge.net/svnroot/brook/trunk brook
      <Hit enter>

3)  Edit the batch file: set_brook_env.bat in GPUIT.
    This batch file sets up for default paths for building brook.  
    You will need to customize it for your environment.

4)  Run a "command window": cmd.exe

5)  Within the command window, run: set_brook_env.bat

6)  Within the command window, run: c:\cygwin\cygwin.bat

7)  Change directory to brook.

8a) Edit config/Windows_NT.mk.

    Remove:
      CFLAGS           += /WX
    Add:
      CFLAGS += /DYY_NO_UNISTD_H
      CFLAGS += /Disatty=_isatty

8b) If you wish to link with MSVCRT instead of LIBCMT, you need to change
    another line.  

    Change this:
       C_RELEASE_FLAG   := /O2 /arch:SSE2 /EHsc /DNDEBUG /Fd$(OBJDIR)\\

    To this:
       C_RELEASE_FLAG   := /MD /O2 /arch:SSE2 /EHsc /DNDEBUG /Fd$(OBJDIR)\\

8c) Edit brcc/ps2arb_lexer.cpp

    Add:
       #include <io.h>

9)  Run make in release mode:
      RELEASE=1 make

10) You will get a error running link.  I suggest you rename the 
    c:\cygwin\bin\link.exe to c:\cygwin\bin\link.cygwin.exe

11) Rerun RELEASE=1 make

    The build will die half way through, but as long as bin/brcc.exe 
    and bin/brook.lib are made properly that is all we care about.

    If you like to build everything, you can run dos2unix on each file 
    that fails.


***** How to compile GPUIT with MSVC 2005 *****

1) Make sure the GPUIT source directory doesn't have any spaces 
   in the path (brook/fxc has problems with these).

2) Make sure that the fxc compiler is in your PATH so that brcc
   can find it.

3) Download and install cmake (www.cmake.org).

4) Run cmake
     Select source directory and binary directory
        Don't use any spaces in the path
     Click configure
     Select makefile format (e.g. MS VC 2005)
     <CMake will configure>
     Set brook directory
     Click configure again
     Click OK

5) Open project file in binary directory (e.g. gpuit.sln)
     Note: If there are no brook projects (e.g. "fdk_brook"), then 
     it means cmake couldn't find the brook executable or libraries.
     You need to fix it and re-run cmake.

6) Change build to "Release".  Debug doesn't work on MSVC 2005. 

7) Click "Build Solution".

8) Click "Build Solution" again.  This is needed because brcc returns 
     an error code for ARB & PS20 targets which stops the build, 
     even though the .cpp is generated correctly.


***** How to compile GPUIT with MSVC Express *****

Before doing the above, you have to add the SDK to your build path.
Fire up the MSVC express GUI, and do the following:

1) Tools -> Options -> Projects and Solutions -> VC++ Directories

2) Add e.g. C:\Program Files\Microsoft Platform SDK for Windows Server 2003 R2\Lib to "Library files

3) Add e.g. C:\Program Files\Microsoft Platform SDK for Windows Server 2003 R2\Include to "Include files"


***** How to compile GPUIT with cygwin/gcc on Windows *****

This works with the reference versions.  Don't know if it can be 
compiled to use brook or not.

1) mkdir /path/to/build/files; cd /path/to/build/files

2) cmake /path/to/gpuit

3) make


***** How to compile GPUIT with MSYS/gcc on Windows *****

This works with the reference versions.  Don't know if it can be 
compiled to use brook or not.

There is a trick to this: you need to run the win32 cmake from 
the command line.

1) mkdir /path/to/build/files; cd /path/to/build/files

2) /c/Program\ Files/CMake\ 2.4/bin/cmake -G"MSYS Makefiles" /path/to/gpuit

3) make


***** How to run GPUIT executables *****

You need to set the brook runtime variables to get GPU acceleration.
Only the DirectX9 backend works.  Using the %COMSPEC% shell, this 
is done like this:

   set BRT_RUNTIME=dx9

