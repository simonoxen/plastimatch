Command Reference
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
The plastimatch adjust command is used to adjust the intensity values 
within an image.  The only operations available are truncation and 
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



plastimatch compare
-------------------

The command line usage is given as follows::

  Usage: plastimatch compare image_in_1 image_in_2

.. _plastimatch_convert:

plastimatch convert
-------------------
The plastimatch convert command is an alias for plastimatch warp.

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


plastimatch diff
----------------

The command line usage is given as follows::

  Usage: plastimatch diff image_in_1 image_in_2 image_out

plastimatch mask
----------------

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

plastimatch stats
-----------------

The command line usage is given as follows::

  Usage: plastimatch stats [options]
  Required:
      --input=image_in

plastimatch warp
----------------
The plastimatch warp command is an alias for plastimatch convert.  
Please refer to :ref:`plastimatch_convert` for usage information.
