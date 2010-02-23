.. plastimatch documentation master file, created by
   sphinx-quickstart on Sat Feb  6 10:25:18 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to plastimatch
======================

Plastimatch is an open source software for deformable image registration. 
It is designed for high-performance
volumetric registration of medical images, such as 
X-ray computed tomography (CT), magnetic resonant imaging (MRI), 
and positron emission tomography (PET). Software features include:

* B-spline method for deformable image registration 
  (GPU and multicore accelerated)
* Demons method for deformable image registration 
  (GPU accelerated)
* ITK-based algorithms for translation, rigid, affine,
  demons, and B-spline registration
* Pipelined, multi-stage registration framework with seamless conversion 
  between most algorithms and transform types
* Landmark-based deformable registration using thin-plate splines for 
  global registration
* Landmark-based deformable registration using radial basis functions 
  for local corrections of deformable registration
* Broad support for 3D image file formats (using ITK), including 
  Dicom, Nifti, NRRD, MetaImage, and Analyze
* Dicom and DicomRT import and export
* XiO import and export
* Plugin to 3D Slicer

Plastimatch also features two handy utilities which are not
directly related to image registration:

* FDK cone-beam CT reconstruction (GPU and multicore accelerated)
* Digitally reconstructed radiograph (DRR) generation
  (GPU and multicore accelerated)

Plastimatch lacks the following:

* Landmark-based rigid registration
* Viscous fluid registration
* FEM registration
* Surface matching registration
* Non-volumetric registration (e.g. 2D-2D or 2D-3D)

For more information:

* `Download page (svn) <http://forge.abcd.harvard.edu/gf/project/plastimatch/scmsvn/?action=AccessInfo>`_
* `License (BSD-style) <http://forge.abcd.harvard.edu/gf/project/plastimatch/scmsvn/?action=browse&path=%2F*checkout*%2Fplastimatch%2Ftrunk%2FLICENSE.TXT&revision=1>`_

Documentation:

.. toctree::
   :maxdepth: 1

   contents

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`search`
