3D Slicer plugin
================

Plastimatch is available as an Extension (plug-in) for 3D Slicer.  
This is one of the easiest methods for using plastimatch, 
because it does not require using the complicated command line interface.  
Furthermore, all of the 3D Slicer Extensions are pre-compiled, and 
easy to download and use.  

The following 3D Slicer binaries have been tested::

  Windows 32-bit
    Slicer3-3.6-2010-06-10-win32.exe                       Works
    Slicer3-3.6.1-2010-08-23-win32.exe                     Doesn't work(1)

  Linux 64-bit
    Slicer3-3.6-2010-06-10-linux-x86_64                    Might work(2)
    Slicer3-3.6.1-2010-08-23-linux-x86_64                  Doesn't work(1)

(1) The extension doesn't show up in the list.  
We are working on a fix for this.

(2) If the linux plugin doesn't work, try installing libg2c.so.  
We are working on a fix for this.

Currently, there three different 3D Slicer Extensions.  They are 
documented on the 3D Slicer web site.  

* `Automatic deformable image registration <http://www.slicer.org/slicerWiki/index.php/Modules:Plastimatch>`_
* Manual deformable image registration (not yet documented)
* `DICOM / DICOM-RT import <http://www.slicer.org/slicerWiki/index.php/Modules:PlastimatchDICOMRT>`_

