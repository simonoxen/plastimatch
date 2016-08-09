Making a tarball
================
This section describes how to create an official packaged version
of plastimatch.

Step 1: Preliminary testing
---------------------------
#. Make sure the changelog is up-to-date
#. Make tarball
     git archive --prefix=plastimatch-1.6.4/ master | bzip2 > ../plastimatch-1.6.4.tar.bz2
#. Unpack and test tarball on linux (don't skip this step)
#. Unpack and test tarball on windows (don't skip this step)

Step 2: Marking the version
---------------------------
#. Bump version number in CHANGELOG
#. Bump version number in CMakeLists
#. Bump version number in doc/sphinx/conf.py
#. Update in remote
#. Tag version

Step 3: Making the final version
--------------------------------
#. Make tarball
#. Unpack and test tarball on linux (don't skip this step)
#. Unpack and test tarball on windows (don't skip this step)
#. Upload to sourceforge
