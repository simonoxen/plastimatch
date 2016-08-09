Making a tarball
================
This section describes how to create an official packaged version
of plastimatch.

#. Update source into plastimatch-pristene
#. Run make package_source
#. Unpack and test tarball (don't skip this step)
#. Reboot and test tarball on windows (don't skip this step)
#. Make sure the changelog is up-to-date
#. Add version number and date to changelog.  This is found in::
     ~/build/plastimatch-pristene/extra_stuff
#. Bump version number in CMakeLists
#. Bump version number in doc/sphinx/conf.py
#. Tag version
#. Upload to sourceforge
