Packaging plastimatch
=====================
This section describes the recommended build configuration for 
building and packaging the official plastimatch tarballs 
and binaries.

Making tarball and debian package
---------------------------------
This is done on wormwood.  

Setting up a build system for the first time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Set DEBEMAIL and DEBFULLNAME environment variables (see http://www.debian.org/doc/manuals/maint-guide/first.en.html)

#. Install the requisite packages::

     sudo apt-get install devscripts pbuilder debhelper

#. Make and register ssh keys::

     ssh-keygen -f ~/.ssh/id_rsa_alioth

   Be sure to set up your ~/.ssh/config file to tell it where to find the key::

     # Add this to ~/.ssh/config
     Host svn.debian.org
             IdentityFile ~/.ssh/id_rsa_alioth

   Then go to https://alioth.debian.org/account/editsshkeys.php to register the public key.  Wait up to one hour for the key to be registered.

#. Download the debian-med repository::

     debcheckout --user <username> svn://svn.debian.org/debian-med/trunk/packages/<package> <package>

#. Link the helper scripts to the debian plastimatch directory::

     cd debian-med/plastimatch
     ln -s ~/work/plastimatch/extra/debian/* .

#. Initial setup of pbuilder environment::

     sudo apt-get install debian-archive-keyring
     sudo pbuilder create --distribution sid --mirror ftp://ftp.us.debian.org/debian/ --debootstrapopts "--keyring=/usr/share/keyrings/debian-archive-keyring.gpg"

   See this link for an explanation https://wiki.ubuntu.com/PbuilderHowto, 
   but use the sid distribution instead of squeeze.


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
#. Bump version number in doc/sphinx/conf.py

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
  DCMTK           3.6.0             (If mondoshot is built)
  FFTW            3.2.2
  ITK             3.20.1
  wxWidgets       2.8.12            (If mondoshot is built)

Configuration settings::

  BUILD_SHARED                  ON      (this is not default)
  PLM_CONFIG_USE_SS_IMAGE_VEC   ON      (this is default)
  PLM_CUDA_ALL_DEVICES          ON      (this is default)
  PLM_INSTALL_RPATH             OFF     (this is default)

#. Build/install all required 3rd party libraries.
#. Double check CPACK version number (at bottom of CMakeLists.txt)
#. Verify that svn is not modified (i.e. do svn update; svn diff)
#. Build plastimatch (start with a fresh cmake)
#. Run test cases, make sure all pass
#. Build package
#. Test package for missing dlls by making sure plastimatch runs

Windows binaries should not include the 3D Slicer plugins.  
Those will be handled by the Slicer extension system.
