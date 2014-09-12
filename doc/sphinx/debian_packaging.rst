Debian packaging
================
This section describes the recommended build configuration for 
building and packaging the official source tarballs 
and debian packages.

Setting up a build system for the first time
--------------------------------------------
#. Set DEBEMAIL and DEBFULLNAME environment variables (see http://www.debian.org/doc/manuals/maint-guide/first.en.html)

#. Install the requisite packages::

     sudo apt-get install devscripts pbuilder debhelper gcc-4.9

   Note: devscripts must be 2.14.2 or higher, gcc must be 4.9 or higher.
   To set up gcc, you might need to do something like this:

      http://lektiondestages.blogspot.com/2013/05/installing-and-switching-gccg-versions.html

#. Make and register ssh keys::

     ssh-keygen -f ~/.ssh/id_rsa_alioth

   Be sure to set up your ~/.ssh/config file to tell it where to find the key::

     # Add this to ~/.ssh/config
     Host svn.debian.org
             IdentityFile ~/.ssh/id_rsa_alioth

   Then go to https://alioth.debian.org/account/editsshkeys.php to register the public key.  Wait up to one hour for the key to be registered.

#. Download the relevant directory from the debian-med repository::

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
---------------------------
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
-------------------------
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
--------------------------------
#. Commit changes to debian files

#. Clean up files from previous version::

     ./clean_directory.sh

#. Repackage tarball::

     cd trunk
     uscan --verbose --force-download

#. Test out by running debuild::

     run_debuild.pl

#. Test out again by running pbuilder::

     run_pbuilder.pl


Rebuilding an existing debian source package
--------------------------------------------
Like this::

 apt-get source foo
 cd foo-0.0.1
 sudo apt-get build-dep foo
 debuild -i -us -uc -b

See: https://wiki.debian.org/HowToPackageForDebian
