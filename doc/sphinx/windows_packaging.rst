Windows packaging
=================
This section describes the recommended build configuration for 
building official plastimatch windows binaries.

As of plastimatch 1.7.0, the following build tools are used::

  VirtualBox      5.1.30            (Debian host)
  MS Windows      10                (Windows 10 Enterprise Evaluation)
  MSVC            15.4.0            (Aka MSVC Community 2017)
  CMake           3.10.1
  WiX             3.11

And the following third party libraries are used::

  CUDA            9.1               (64 bit only)
  DCMTK           3.6.2             (Build separate 32 & 64 bit)
  FFTW            3.3.5             (Download separate 32 & 64 bit)
  ITK             4.13.0            (Build separate 32 & 64 bit)

NVIDIA deprecated support for 32-bit CUDA as of version 7.0.
Plastimatch does not offer that option for binary download.

#. Build DCMTK

   #. Use fresh build directory
   #. Set BUILD_APPS to OFF
   #. Set BUILD_SHARED_LIBS to OFF
   #. Set DCMTK_OVERWRITE_WIN32_COMPILE_FLAGS to OFF

#. Build ITK
   
   #. Use fresh build directory
   #. Set BUILD_SHARED_LIBS to ON
   #. Set Module_ITKReview to ON

#. Build plastimatch

   #. Use fresh build directory
   #. Set GIT_EXECUTABLE to "wsl git"
   #. Set BUILD_SHARED to ON
   #. For 32-bit build, set PLM_CONFIG_ENABLE_CUDA to OFF
   #. For 64-bit build, set PLM_CUDA_ALL_DEVICES to ON
   #. For 64-bit build, set CUDA_HOST_COMPILER manually
      See: https://github.com/opencv/opencv/issues/9908

#. Run test cases, make sure all pass
#. Build PACKAGE
#. Test package on fresh VM
