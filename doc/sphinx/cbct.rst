FDK - Cone-beam reconstruction
==============================
The term FDK refers to the authors 
Feldkamp, Davis, and Kress who wrote the seminal paper 
"Practical cone-beam algorithm" in 1984.  Their paper 
describes a filtered back-projection reconstruction algorithm 
for cone-beam geometries.  The fdk program in plastimatch is 
an implmenetation of the FDK algorithm.

FDK usage
---------
The fdk program takes a directory of 2D projection images as input, and 
generates a single 3D volume as output.  

The command line usage is::

  Usage: fdk [options]
  Options:
   -A hardware            Either "cpu" or "brook" or "cuda" (default=cpu)
   -a "num ((num) num)"   Use this range of images
   -r "r1 r2 r3"          Set output resolution (in voxels)
   -f filter              Either "none" or "ramp" (default=ramp)
   -s scale               Scale the intensity of the output file
   -z "s1 s2 s3"          Physical size of the reconstruction (in mm)
   -I indir               The input directory
   -O outfile             The output file
   -sb ". (default)" The subfolder with *.raw files
   -F {F,H}               Full or Half fan correction
   -cor                   Turn on Coronal output
   -sag                   Turn on Sagittal output

Image geometry
--------------
By default, when you generate a DRR, the image is oriented as if the
virtual x-ray source were a camera.  That means that for a right
lateral film, the columns of the image go from inf to sup, and the
rows go from ant to post.  The Varian OBI system produces HND files,
which are oriented differently. For a right lateral film, the columns
of the HND images go from ant to post, and the rows go from sup to
inf.  An illustration of this idea is shown in the figure below. 

.. figure:: ../figures/cbct_geometry.png
   :width: 60 %

   Geometry of Varian HND files
