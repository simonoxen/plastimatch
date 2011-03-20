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
  plastimatch version 1.4-beta (2161)
  Usage: plastimatch command [options]
  Commands:
    add           adjust        autolabel     crop          compare     
    compose       convert       diff          dvh           header      
    mask          register      resample      segment       stats       
    thumbnail     warp          xio-dvh     

  For detailed usage of a specific command, type:
    plastimatch command

plastimatch add
---------------
The *add* command is used to add one or more images together and create 
an output image.

The command line usage is given as follows::

  Usage: plastimatch add input_file [input_file ...] output_file

Example
^^^^^^^
To add together files 01.mha, 02.mha and 03.mha, and save the result 
in the file output.mha, you can run the following command::

  plastimatch add 01.mha 02.mha 03.mha output.mha

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
      --output-type={uchar,short,ushort,ulong,float}
      --scale="min max"
      --ab-scale="ab nfx ndf"       (Alpha-beta scaling)
      --stretch="min max"
      --truncate-above=value
      --truncate-below=value

Example
^^^^^^^
The following command will truncate the input intensities to the 
range [-1000,1000], and then map the intensities to the range [0,1]::

  plastimatch adjust --input infile.nrrd --output outfile.nrrd \
    --truncate-above 1000 --truncate-below -1000 \
    --stretch "0 1"

plastimatch autolabel
---------------------
The *autolabel* command is an experimental program the uses machine 
learning to identify the thoracic vertibrae in a CT scan.  

The command line usage is given as follows::

  Usage: plastimatch autolabel [options]
  Options:
    -h, --help            Display this help message 
        --input <arg>     Input image filename (required) 
        --network <arg>   Input trained network filename (required) 
        --output <arg>    Output csv filename (required) 

plastimatch crop
----------------
The *crop* command crops out a rectangular portion of the input file, 
and saves that portion to an output file.
The command line usage is given as follows::

  Usage: plastimatch crop [options]
  Required:
      --input=image_in
      --output=image_out
      --voxels="x-min x-max y-min y-max z-min z-max" (integers)

The voxels are indexed starting at zero.
In other words, if the size of the image is 
:math:`M \times N \times P`,
the x values should range between 0 and :math:`M-1`.

Example
^^^^^^^
The following command selects the region of size 
:math:`10 \times 10 \times 10`, with the first voxel of the output 
image being at location (5,8,12) of the input image::

  plastimatch crop --input in.mha --output out.mha --voxels "5 14 8 17 12 21"

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

plastimatch compose
-------------------
The *compose* command is used to compose two transforms.  
The command line usage is given as follows::

  Usage: plastimatch compose file_1 file_2 outfile

  Note:  file_1 is applied first, and then file_2.
            outfile = file_2 o file_1
            x -> x + file_2(x + file_1(x))

The transforms can be of any type, including translation, rigid, affine, 
itk B-spline, native B-spline, or vector fields.  
The output file is always a vector field.  

There is a further restriction that at least one of the input files 
must be either a native B-spline or vector field.  This restriction 
is required because that is how the resolution and voxel spacing 
of the output vector field is chosen.

Example
^^^^^^^
Suppose we want to compose a rigid transform (rigid.tfm) with a vector field
(vf.mha), such that the output transform is equivalent to applying 
the rigid transform first, and the vector field second.
::
  platimatch compose rigid.tfm vf.mha composed_vf.mha

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
      --patient-pos={hfs,hfp,ffs,ffp}
      --dicom-dir=directory      (for structure association)
      --ctatts=filename          (for dij)
      --dif=filename             (for dij)
      --input-ss-img=filename    (for structures)
      --input-ss-list=filename   (for structures)
      --prune-empty              (for structures)
      --input-dose-img=filename  (for rt dose)
      --input-dose-xio=filename  (for XiO rt dose)
      --input-dose-ast=filename  (for Astroid rt dose)
      --input-dose-mc=filename   (for Monte Carlo 3ddose rt dose)

      --output-cxt=filename      (for structures)
      --output-dicom=directory   (for image and structures)
      --output-dij=filename      (for dij)
      --output-dose-img          (for rt dose)
      --output-img=filename      (for image)
      --output-labelmap=filename (for structures)
      --output-prefix=string     (for structures)
      --output-ss-img=filename   (for structures)
      --output-ss-list=filename  (for structures)
      --output-vf=filename       (for vector field)
      --output-xio=directory     (for rt dose and structures)

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
The plastimatch *diff* command subtracts one image from another, and saves 
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

plastimatch dvh
---------------
The *dvh* command creates a dose value histogram (DVH) 
from a given dose image and structure set image.  
The command line usage is given as follows::

  Usage: plastimatch dvh [options]
     --input-ss-img file
     --input-ss-list file
     --input-dose file
     --output-csv file
     --input-units {gy,cgy}
     --cumulative
     --num-bins
     --bin-width

The required inputs are 
--input-dose, 
--input-ss-img, --input-ss-list, 
and --output-csv.
The units of the input dose must be either Gy or cGy.  
DVH bin values will be generated for all structures found in the 
structure set files.  The output will be generated as an ASCII 
csv-format spreadsheet file, readable by OpenOffice.org or Microsoft Excel.

The default is a differential (standard) histogram, rather than the 
cumulative DVH which is most common in radiotherapy.  To create a cumulative 
DVH, use the --cumulative option.  

The default is to create 256 bins, each with a width of 1 Gy.  
You can adjust these values using the --num-bins and --bin-width option.

Example
^^^^^^^
To generate a DVH for a single 2 Gy fraction, we might choose 250 bins each of 
width 1 cGy.  If the input dose is already specified in cGy, you would 
use the following command::

  plastimatch dvh \
    --input-ss-img structures.mha \
    --input-ss-list structures.txt \
    --input-dose dose.mha \
    --output-csv dvh.csv \
    --input-units cgy \
    --num-bins 250 \
    --bin-width 1

plastimatch header
------------------
The *header* command displays brief information about the image geometry.
The command line usage is given as follows::

  Usage: plastimatch header input-file


Example
^^^^^^^
We can display the geometry of any supported file type, such as mha, nrrd, 
or dicom.  We can run the command as follows::

  $ plastimatch header input.mha
  Origin = -180 -180 -167.75
  Size = 512 512 120
  Spacing = 0.7031 0.7031 2.5
  Direction = 1 0 0 0 1 0 0 0 1

From the header information, we see that the image has 120 slices, 
and each slice is 512 x 512 pixels.  The slice spacing is 2.5 mm, 
and the in-plane pixel spacing is 0.7031 mm.

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
The plastimatch *register* command is used to peform linear or deformable 
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
              --fixed=file
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

plastimatch segment
-------------------
The *segment* command does simple threshold-based semgentation.  
The command line usage is given as follows::

  Usage: plastimatch segment [options]
  Options:
    -h, --help                    Display this help message 
        --input <arg>             Input image filename (required) 
        --lower-threshold <arg>   Lower threshold (include voxels above this 
                                   value) 
        --output-dicom <arg>      Output dicom directory (for RTSTRUCT) 
        --output-img <arg>        Output image filename 
        --upper-threshold <arg>   Upper threshold (include voxels below this 
                                   value) 

Example
^^^^^^^
Suppose we have a CT image of a water tank, and we wish to create an image 
which has ones where there is water, and zeros where there is air.  
Then we could do this::

  plastimatch segment \
    --input water.mha \
    --output-img water-label.mha \
    --lower-threshold -500

If we wanted instead to create a DICOM-RT structure set, we should 
specify a DICOM image as the input.  This will allow plastimatch to 
create the DICOM-RT with the correct patient name, patient id, and UIDs.
The output file will be called "ss.dcm".
::
  plastimatch segment \
    --input water_dicom \
    --output-dicom water_dicom \
    --lower-threshold -500

plastimatch stats
-----------------
The plastimatch stats command displays a few basic statistics about the 
image onto the screen.

The command line usage is given as follows::

  Usage: plastimatch stats file [file ...]

The input files can be either 2D projection images, 3D volumes, or 
3D vector fields.

Example
^^^^^^^
The following command displays statistics for the 3D volume synth_1.mha. ::

  $ plastimatch stats synth_1.mha
  MIN -999.915161 AVE -878.686035 MAX 0.000000 NUM 54872

The reported statistics are interpreted as follows::

  MIN      Minimum intensity in image
  AVE      Average intensity in image
  MAX      Maximum intensity in image
  NUM      Number of voxels in image

Example
^^^^^^^
The following command displays statistics for the 3D vector field vf.mha::

  $ plastimatch stats vf.mha
  Min:            0.000     -0.119     -0.119
  Mean:          13.200      0.593      0.593
  Max:           21.250      1.488      1.488
  Mean abs:      13.200      0.594      0.594
  Energy: MINDIL -6.79753 MAXDIL 0.166026 MAXSTRAIN 41.5765 TOTSTRAIN 70849.7
  Min dilation at: (29 19 19)
  Jacobian: MINJAC -6.32835 MAXJAC 1.15443 MINABSJAC 0.360538
  Min abs jacobian at: (28 36 36)
  Second derivatives: MINSECDER 0 MAXSECDER 388.821 TOTSECDER 669219 INTSECDER 1.5245e+06
  Max second derivative: (29 36 36)

The rows corresponding to "Min, Mean, Max, and Mean abs" each 
have three numbers, which correspond to the x, y, and z coordinates.  
Therefore, they compute these statistics for each vector direction 
separately.

The remaining statistics are described as follows::

  MINDIL        Minimum dilation
  MAXDIL        Maximum dilation
  MAXSTRAIN     Maximum strain
  TOTSTRAIN     Total strain
  MINJAC        Minimum Jacobian     
  MAXJAC        Maximum Jacobian
  MINABSJAC     Minimum absolute Jacobian
  MINSECDER     Minimum second derivative
  MAXSECDER     Maximum second derivative
  TOTSECDER     Total second derivative
  INTSECDER     Integral second derivative

plastimatch thumbnail
---------------------
The *thumbnail* command generates a two-dimensional thumbnail image of an 
axial slice of the input volume.  The output image 
is not required to correspond exactly to an integer slice number.  
The location of the output image within the slice is always centered. 

The command line usage is given as follows::

  Usage: plastimatch thumbnail [options] input-file
  Options:
    --input file
    --output file
    --thumbnail-dim size
    --thumbnail-spacing size
    --slice-loc location

Example
^^^^^^^
We create a two-dimensional image with resolution 10 x 10 pixels,
at axial location 0, and of size 20 x 20 mm::

  plastimatch thumbnail \
    --input in.mha --output out.mha \
    --thumbnail-dim 10 \
    --thumbnail-spacing 2 \
    --slice-loc 0

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
