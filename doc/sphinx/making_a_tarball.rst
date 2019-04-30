.. _making_a_tarball:

Making a tarball
================
This section describes how to create an official packaged version
of plastimatch.

Step 1: Preliminary testing
---------------------------
#. Download tarball from gitlab::

     wget https://gitlab.com/plastimatch/plastimatch/-/archive/master/plastimatch-master.tar.gz

#. Unpack and test tarball on linux (don't skip this step)
#. Unpack and test tarball on windows (don't skip this step)
#. Unpack and test tarball on mac (don't skip this step)
#. Test parallel regression tests (don't skip this step)::

     cd ~/build/plastimatch
     ctest -j 16

Step 2: Marking the version
---------------------------
#. Update CHANGELOG; Bump version number in CHANGELOG
#. Bump version number in CMakeLists
#. Bump version number in doc/sphinx/conf.py
#. Also check copyright year range in doc/sphinx/conf.py
#. Bump version number in doc/sphinx/plastimatch.rst
#. Regenerate man pages::

     sphinx-build -b man -d ~/shared/web-plastimatch/.doctrees  ~/work/plastimatch/doc/sphinx ~/work/plastimatch/doc/man

#. Push above changes to remote and tag version::

     export V=1.8.0
     git commit -am "Version ${V}"
     git push origin master
     git tag -a "${V}" -m "Version ${V}"
     git push origin --tags

#. Edit changelog on gitlab site.

Step 3: Making the final version
--------------------------------
#. Download tarball from gitlab.
#. Unpack and test tarball on linux (don't skip this step).
#. Unpack and test tarball on windows (don't skip this step).
#. Upload to sourceforge::

     sftp gregsharp@frs.sourceforge.net
     cd /home/pfs/project/plastimatch/Source
     put plastimatch-1.6.4.tar.bz2
