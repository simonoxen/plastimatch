Windows packaging
=================
This section describes the recommended build configuration for 
building official plastimatch windows binaries.

The Windows build uses the MSVC 2013 express compiler.  
This compiler is capable of both 32-bit and 64-bit targets, 
and also supports OpenMP.

As of plastimatch 1.7.0, the following build tools are used::

  VirtualBox      5.1.30            (Debian host)
  MS Windows      10                (Windows 10 Enterprise Evaluation)
  MSVC            15.4.0            (Aka MSVC Community 2017)
  CMake           3.9.5
  WiX             3.11

And the following third party libraries are used::

  CUDA            9.1               (64 bit only)
  DCMTK           3.6.2             (Build separate 32 & 64 bit)
  FFTW            3.3.5             (Download separate 32 & 64 bit)
  ITK             4.13.0            (Build separate 32 & 64 bit)

NVIDIA deprecated support for 32-bit CUDA as of version 7.0.
This is why plastimatch does not offer that option for binary download.

Configuration settings::

  BUILD_SHARED                  ON      (this is not default)
  PLM_CUDA_ALL_DEVICES          ON      (this is not default)

#. Build/install ITK and DCMTK

   #. DCMTK is built as static libraries
   #. ITK is built as shared libraries

#. Double check CPACK version number (at bottom of CMakeLists.txt)
#. Build plastimatch (start with a fresh build directory)

   #. Set CUDA_HOST_COMPILER manually
      See: https://github.com/opencv/opencv/issues/9908

#. Run test cases, make sure all pass
#. Build package
#. Test package for missing dlls, etc.
