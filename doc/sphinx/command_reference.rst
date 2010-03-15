Command reference
=================
The plastimatch executable is used for 
a variety of operations, including image
registration, image warping, image resampling, and file format
conversion.  The general format of the command is::

  plastimatch command [options]

where the form of the options depends upon the command given.
The list of possible commands can be seen by simply typing "plastimatch" 
without any additional command line arguments::

  $ plastimatch
  plastimatch version 1.4-beta (1008)
  Usage: plastimatch command [options]
  Commands:
    adjust        compare     
    convert       diff        
    mask          register    
    resample      stats       
    warp        
  For detailed usage of a specific command, type:
    plastimatch command

plastimatch adjust
------------------
The *adjust* command is used to adjust the intensity values 
within an image.  The adjustment operations available are truncation and 
linear scaling.  

The command line usage is given as follows::

  Usage: plastimatch adjust [options]
  Required:
      --input=image_in
      --output=image_out
  Optional:
      --truncate-above=value
      --truncate-below=value
      --stretch="min max"
      --output-type={uchar,short,ushort,ulong,float}

Example
^^^^^^^
The following command will truncate the input intensities to the 
range [-1000,1000], and then map the intensities to the range [0,1]::

  plastimatch adjust --input infile.nrrd --output outfile.nrrd \
    --truncate-above 1000 --truncate-below -1000 \
    --stretch "0 1"

plastimatch compare
-------------------
The *compare* command compares two files by subtracting 
one file from the other, and reporting statistics 
of the difference image.
The two input files must have the 
same geometry (origin, dimensions, and voxel spacing).
The command line usage is given as follows::

  Usage: plastimatch compare image_in_1 image_in_2

Example
^^^^^^^
The following command subtracts synth_2 from synth_1, and 
reports the statistics::

  $ plastimatch compare synth_1.mha synth_2.mha 
  MIN -558.201904 AVE 7.769664 MAX 558.680847
  MAE 85.100204 MSE 18945.892578
  DIF 54872 NUM 54872

The reported statistics are interpreted as follows::

  MIN      Minimum value of difference image
  AVE      Average value of difference image
  MAX      Maximum value of difference image
  MAE      Mean average value of difference image
  MSE      Mean squared difference between images
  DIF      Number of pixels with different intensities
  NUM      Total number of voxels in the difference image

.. _plastimatch_convert:

plastimatch convert
-------------------
The *convert* command is used to convert files from one 
format to another format.  As part of the conversion process, it can 
also apply (linear or deformable) geometric transforms 
to the input images.  In fact, *convert* is just an alias for the 
*warp* command.

The command line usage is given as follows::

  Usage: plastimatch convert [options]
  Options:
      --input=filename
      --xf=filename
      --interpolation=nn
      --fixed=filename
      --offset="x y z"
      --spacing="x y z"
      --dims="x y z"
      --default-val=number
      --output-type={uchar,short,float,...}
      --algorithm=itk
      --dicom-dir=directory      (for structure association)
      --ctatts=filename          (for dij)
      --dif=filename             (for dij)
      --input-ss-img=filename    (for structures)
      --prune-empty              (for structures)

      --output-cxt=filename      (for structures)
      --output-dicom=directory   (for image and structures)
      --output-dij=filename      (for dij)
      --output-img=filename      (for image)
      --output-labelmap=filename (for structures)
      --output-prefix=string     (for structures)
      --output-ss-img=filename   (for structures)
      --output-ss-list=filename  (for structures)
      --output-vf=filename       (for vector field)
      --output-xio=directory     (for structures)

Examples
^^^^^^^^
The first example demonstrates how to convert 
a DICOM volume to NRRD.  The DICOM images 
that comprise the volume must be 
stored in a single directory, which for this example 
is called "dicom-in-dir".  Because the --output-type option was 
not specified, 
the output type will be matched to the type of the input DICOM volume. 
The format of the output file (NRRD) is determined from the filename 
extension. ::

  plastimatch convert --input dicom-in-dir --output outfile.nrrd

This example further converts the type of the image intensities to float. ::

  plastimatch convert --input dicom-in-dir --output outfile.nrrd \
    --output-type float

The next example shows how to resample the output image to a different 
geometry.  The --offset option sets the position of the 
(center of) the first voxel of the image, the --dim option sets the 
number of voxels, and the --spacing option sets the 
distance between voxels.  The units for offset and spacing are 
assumed to be millimeters. ::

  plastimatch convert --input dicom-in-dir --output outfile.nrrd \
    --offset "-200 -200 -165" \
    --dim "250 250 110" \
    --spacing "2 2 2.5"

Generally speaking, it is tedious to manually specify the geometry of 
the output file.  If you want to match the geometry of the output 
file with an existing file, you can do this using the --fixed option. ::

  plastimatch convert --input dicom-in-dir --output outfile.nrrd \
    --fixed reference.nrrd

This next example shows how to convert a DICOM RT structure set file 
into an image using the --output-ss-img option.  
Because structures in DICOM RT are polylines, they are rasterized to 
create the image.  The voxels of the output image are 32-bit integers, 
where the i^th bit of each integer has value one if the voxel lies with 
in the corresponding structure, and value zero if the voxel lies outside the
structure.  The structure names are stored in separate file using 
the --output-ss-list option. ::

  plastimatch convert --input structures.dcm \
    --output-ss-img outfile.nrrd \
    --output-ss-list outfile.txt

In the previous example, the geometry of the output file wasn't specified.
When the geometry of a DICOM RT structure set isn't specified, it is 
assumed to match the geometry of the DICOM CT image associated with the 
contours.  If the associated DICOM CT image is in the same directory as 
the structure set file, it will be found automatically.  Otherwise, we 
have to tell plastimatch where it is located with the --dicom-dir option. ::

  plastimatch convert --input structures.dcm \
    --output-ss-img outfile.nrrd \
    --output-ss-list outfile.txt \
    --dicom-dir ../ct-directory


plastimatch diff
----------------
The plastimatch diff command subtracts one image from another, and saves 
the output as a new image.
The two input files must have the 
same geometry (origin, dimensions, and voxel spacing).

The command line usage is given as follows::

  Usage: plastimatch diff image_in_1 image_in_2 image_out

Example
^^^^^^^
The following command computes file1.nrrd minus file2.nrrd, and saves 
the result in outfile.nrrd::

  plastimatch diff file1.nrrd file2.nrrd outfile.nrrd

plastimatch mask
----------------
The *mask* command is used to fill in a region of the image, as specified
by a mask file, with a constant intensity.  

The command line usage is given as follows::

  Usage: plastimatch mask [options]
  Required:
      --input=image_in
      --output=image_out
      --mask=mask_image_in
  Optional:
      --negate-mask
      --mask-value=float
      --output-format=dicom
      --output-type={uchar,short,ushort,ulong,float}

Examples
^^^^^^^^
If we have a file prostate.nrrd which is non-zero inside of the prostate 
and zero outside of the prostate, we can set the prostate intensity to 1000
(while leaving non-prostate areas with their original intensity) using 
the following command. ::

  plastimatch mask \
    --input infile.nrrd \
    --output outfile.nrrd \
    --mask-value 1000 \
    --mask prostate.nrrd

Suppose we have a file called patient.nrrd, which is non-zero inside of the 
patient, and zero outside of the patient.  If we want to fill in the area 
outside of the patient with value -1000, we use the following command. ::

  plastimatch mask \
    --input infile.nrrd \
    --output outfile.nrrd \
    --negate-mask \
    --mask-value 1000 \
    --mask patient.nrrd

plastimatch register
--------------------
The plastimatch register command is used to peform linear or deformable 
registration of two images.  
The command line usage is given as follows::

  Usage: plastimatch register command_file

A more complete description, including the format of the required 
command file is given in the next section.

plastimatch resample
--------------------
The *resample* command can be used to change the geometry of an image.

The command line usage is given as follows::

  Usage: plastimatch resample [options]
  Required:   --input=file
              --output=file
  Optional:   --subsample="x y z"
              --origin="x y z"
              --spacing="x y z"
              --size="x y z"
              --output_type={uchar,short,ushort,float,vf}
              --interpolation={nn, linear}
              --default_val=val

Example
^^^^^^^
We can use the --subsample option to bin an integer number of voxels 
to a single voxel.  So for example, if we want to bin a cube of size 
3x3x1 voxels to a single voxel, we would do the following. ::

  plastimatch resample \
    --input infile.nrrd \
    --output outfile.nrrd \
    --subsample "3 3 1"

plastimatch stats
-----------------
The plastimatch stats command displays a few basic statistics about the 
image onto the screen.

The command line usage is given as follows::

  Usage: plastimatch stats [options]
  Required:
      --input=image_in

Example
^^^^^^^
The following command displays statistics for the file synth_1.mha. ::

  $ plastimatch stats --input synth_1.mha
  MIN -999.915161 AVE -878.686035 MAX 0.000000 NUM 54872

The reported statistics are interpreted as follows::

  MIN      Minimum intensity in image
  AVE      Average intensity in image
  MAX      Maximum intensity in image
  NUM      Number of voxels in image

plastimatch warp
----------------
The *warp* command is an alias for *convert*.  
Please refer to :ref:`plastimatch_convert` for the list of command line 
parameters.

Examples
^^^^^^^^
To warp an image using the B-spline coefficients generated by the 
plastimatch register command (saved in the file bspline.txt), do the 
following::

  plastimatch warp \
    --input infile.nrrd \
    --output outfile.nrrd \
    --xf bspline.txt

In the previous example, the output file geometry was determined by the 
geometry information in the bspline coefficient file.  You can resample 
to a different geometry using --fixed, or --origin, --dim, and --spacing. ::

  plastimatch warp \
    --input infile.nrrd \
    --output outfile.nrrd \
    --xf bspline.txt \
    --fixed reference.nrrd

When warping a structure set image, where the integer bits correspond to 
structure membership, you need to use nearest neighbor interpolation 
rather than linear interpolation. ::

  plastimatch warp \
    --input structures-in.nrrd \
    --output structures-out.nrrd \
    --xf bspline.txt \
    --interpolation nn

Sometimes, voxels located outside of the geometry of the input image 
will be warped into the geometry of the output image.  By default, these 
areas are "filled in" with an intensity of zero.  You can choose a different 
value for these areas using the --default-val option. ::

  plastimatch warp \
    --input infile.nrrd \
    --output outfile.nrrd \
    --xf bspline.txt \
    --default-val -1000


