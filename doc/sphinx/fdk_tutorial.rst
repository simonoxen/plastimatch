.. _fdk_tutorial:

FDK Tutorial
============
This tutorial demonstrates the use of the FDK program.  It is highly 
recommended to try this tutorial before attempting to reconstruct your 
own images.  

The basic procedure we will follow is to generate synthetic X-ray projection 
images using the drr program, and then reconstruct them using the fdk 
program.  

Download the sample data
^^^^^^^^^^^^^^^^^^^^^^^^
http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1018/headphantom.mha.zip

The sample data is a 3D volume of a CT scan of an acrylic head phantom. 
You can see what the original image looks below.

.. image:: ../figures/fdk_tutorial_1.png
   :width: 40 %

Make DRRs (first try)
^^^^^^^^^^^^^^^^^^^^^
Run the following command to create drr images::

  drr \
      -t pfm \
      -a 60 \
      -N 3 \
      -g "1000 1500" \
      -r "100 100" \
      -z "300 300" \
      -I headphantom.mha \
      -O head/

Note that the "/" at the end of the output directory "head" is important.
If you forget this, the projection files will be created in the current 
directory.  What you should get is a bunch of files in a directory 
like this::

  head/0000.pfm
  head/0000.txt
  head/0001.pfm
  head/0001.txt
  ...

Reconstruct the image (first try)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the following command to reconstruct the image::

  fdk -f none -r "100 100 100" -I head -O out.mha

You should get an image like this:

.. image:: ../figures/fdk_tutorial_2.png
   :width: 40 %

The image is kind of blurry, which is because we didn't use the ramp filter.
Try again with the ramp filter::

  fdk -f ramp -r "100 100 100" -I head -O out.mha

You should get an image like this:

.. image:: ../figures/fdk_tutorial_3.png
   :width: 40 %

The artifact at the posterior of the skull is a truncation artifact, which 
is caused by missing data in the DRRs that we generated in the previous 
step.  So let's work on it, and get rid of it.

Make DRRs (second try)
^^^^^^^^^^^^^^^^^^^^^^
In order to get better control over the DRR generation process, ...
