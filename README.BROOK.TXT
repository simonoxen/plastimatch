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
