.. _making_a_tarball:

Making a tarball
================
This section describes how to create an official packaged version
of plastimatch.

Step 1: Preliminary testing
---------------------------
#. Download tarball from gitlab.
#. Unpack and test tarball on linux (don't skip this step)
#. Unpack and test tarball on windows (don't skip this step)
#. Test parallel regression tests (don't skip this step)::

     cd ~/build/plastimatch
     ctest -j 16

#. Run lintian on package::

     lintian -i *.changes

Step 2: Marking the version
---------------------------
#. Update CHANGELOG; Bump version number in CHANGELOG
#. Bump version number in CMakeLists
#. Bump version number in doc/sphinx/conf.py
#. Bump version number in doc/sphinx/plastimatch.rst
#. Regenerate man pages::

     sphinx-build -b man -d ~/work/web-plastimatch/.doctrees  ~/work/plastimatch/doc/sphinx ~/work/plastimatch/doc/man

#. Push above changes to remote
#. Tag version::

     git tag -a "v1.6.5" -m "Version 1.6.5"
     git push origin --tags

#. Edit changelog on gitlab site.

Step 3: Making the final version
--------------------------------
#. Download tarball from gitlab.
#. Repackage the tarball from hash-based to version-based.
#. Unpack and test tarball on linux (don't skip this step).
#. Unpack and test tarball on windows (don't skip this step).
#. Upload to sourceforge::

     sftp gregsharp,plastimatch@frs.sourceforge.net
     cd /home/pfs/project/p/pl/plastimatch/Source
     put plastimatch-1.6.4.tar.bz2
