Packaging plastimatch
=====================

This section describes the recommended build configuration for 
building and packaging the official plastimatch binaries.

Plastimatch binaries should not include the 3D Slicer plugins.  
Those will be handled by the Slicer extension system.

Third party libraries to be used (Plastimatch 1.5.0)::

  CUDA            3.0.14
  DCMTK           3.5.4             (Windows only)
  FFTW            3.2.2             (Windows only)
  ITK             3.20.0
  wxWidgets       2.8.12            (Windows only)

Configuration settings - leave at default unless otherwise specified::

  PLM_INSTALL_RPATH           OFF   (Unix only)
  PLM_USE_GPU_PLUGINS         ON
  PLM_CONFIG_VERSION_STRING   1.5.0

Building a windows binary
-------------------------

The Windows build uses the MSVC 2008 express compiler.  
This means 32-bit (only), and no OpenMP.

#. Build/install all required 3rd party libraries.
#. Build plastimatch (start with a fresh cmake)

Building a plastimatch deb package using cpack
----------------------------------------------

This “How to” describes the way for build and 
sign a deb package of plastimatch using the cpack tool.
The deb files that are on the website are builded 
into two GNU/Linux Ubuntu 10.04.1 OSs (32 and
64 bits) virtualized by Virtualbox in a GNU/Linux Fedora 14 64 bits.
This guide is related at the revision number 1930 of plastimatch.

Packages needed::

  gengetopt
  uuid-dev
  gfortran
  sqlite3
  libsqlite3-dev
  dpkg-sig
  build-essential
  cmake

*Dependences*

The only one dependence for the packages that will be builded 
is libgfortran3, but it'll be manage 
from the manager package system of the OS.

*Build the package*

- Download the code via svn and choose the plastimatch folder as working directory
- Create a folder called “build”
- Copy the man page files (locate in plastimatch/doc/man) into the working directory
- Copy the bash completion file (locate in plastimatch/extra/bash_completion) into the working directory
- Choose the folder “build” as working directory and run::

    cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release ..

- And then::

    cpack ..

- wait a few of minuts for the creation of deb file
- finally use dpkg-sig tool for sign the deb file (a gpg key must be alredy created)::

    dpkg-sig --sign USERNAME plastimatch_1930_ARCH_TYPE.deb

For questions, clarifications, corrections or comments please write to:
p.zaffino@yahoo.it
