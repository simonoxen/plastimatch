Packaging plastimatch
=====================
This section describes the recommended build configuration for 
building and packaging the official plastimatch tarballs 
and binaries.

Making tarball and debian package
---------------------------------
This is done on wormwood.  

Step 1: Preliminary testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The preliminary testing will make sure that the tarball will 
build under debian in step 3.

#. Update changelog (in an terminal, not emacs)::

     cd plastimatch/trunk
     dch -v 1.5.4+dfsg0-1

#. Run rebundle.pl until satisfied::

     rebundle.pl

#. Refresh your pbuilder environment (if needed)::

     sudo pbuilder --clean && sudo pbuilder --update

#. Test out by running debuild::

     run_debuild.pl

#. Test out again by running pbuilder::

     run_pbuilder.pl

#. Test parallel regression tests::

      cd ~/build/plastimatch-3.20.0
      ctest -j 4

Step 2: Build the tarball
^^^^^^^^^^^^^^^^^^^^^^^^^

#. Make sure the changelog is up-to-date
#. Update source into plastimatch-pristene
#. Run make package_source
#. Unpack and test tarball (don't skip this step)
#. Reboot and test tarball on windows (don't skip this step)
#. Upload to web site

Then, do a few small things to get ready for next time

#. Add version number and date to changelog.  This is found in::

     ~/build/plastimatch-pristene/extra_stuff

#. Bump version number in CMakeLists

Step 3: Build the debian package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Commit changes to debian files

#. Clean up files from previous version::

     ./clean_directory.sh

#. Repackage tarball::

     cd trunk
     ./debian/get-orig-source

#. Test out by running debuild::

     run_debuild.pl

#. Test out again by running pbuilder::

     run_pbuilder.pl

Building a windows binary
-------------------------
The Windows build uses the MSVC 2008 express compiler.  
This means 32-bit (only), and no OpenMP.

Third party libraries to be used::

  CUDA            3.0.14
  DCMTK           3.6.0             (Nb. Mondoshot used 3.5.4)
  FFTW            3.2.2
  ITK             3.20.1
  wxWidgets       2.8.12            (If mondoshot is built)

Configuration settings::

  PLM_CUDA_ALL_DEVICES        ON      (this is default)
  PLM_INSTALL_RPATH           OFF     (change this, only relevant for Unix)
  PLM_USE_GPU_PLUGINS         ON      (this is default)
  PLM_USE_SS_IMAGE_VEC        ON      (change this, but should be OFF for slicer plugin)

#. Build/install all required 3rd party libraries.
#. Double check CPACK version number (at bottom of CMakeLists.txt)
#. Verify that svn is not modified (i.e. do svn update; svn diff)
#. Build plastimatch (start with a fresh cmake)
#. Run test cases, make sure all pass
#. Build package
#. Test package for missing dlls by making sure plastimatch runs

Windows binaries should not include the 3D Slicer plugins.  
Those will be handled by the Slicer extension system.
