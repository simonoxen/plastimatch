Packaging Plastimatch
---------------------

This section describes the recommended build configuration for packaging 
the plastimatch binaries.

Third party libraries to be included::

  CUDA            3.0
  DCMTK           3.5.4             (Windows only)
  FFTW            latest version
  ITK             3.20.0
  wxWidgets       latest version    (Windows only)
  Slicer          latest version

Configuration settings - leave at default unless otherwise specified::

  PLM_INSTALL_RPATH           ?? (Unix only)
  PLM_USE_GPU_PLUGINS         ON
