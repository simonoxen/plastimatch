Making a tarball
================
This section describes how to create an official packaged version
of plastimatch.

Step 1: Preliminary testing
---------------------------
#. Make sure the changelog is up-to-date
#. Get pristine copy of source::
     git archive --prefix=plastimatch-1.6.4/ master | bzip2 > ../plastimatch-1.6.4.tar.bz2
#. Unpack and test tarball (don't skip this step)
#. Reboot and test tarball on windows (don't skip this step)

Step 2: Marking the version
---------------------------
#. Bump version number in 
#. Bump version number in CMakeLists
#. Bump version number in doc/sphinx/conf.py
#. Update in remote
#. Tag version

Step 3: Making the final version
--------------------------------
#. Make tarball
#. Upload to sourceforge
