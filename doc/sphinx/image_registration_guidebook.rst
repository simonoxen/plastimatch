.. _image_registration_guidebook:

Image registration guidebook
============================

Quick start guide
-----------------

You must create a command file to do registration.  

.. include:: image_registration_quick_start.rst

For more examples, read on!

3-DOF registration (translation)
--------------------------------
Sometimes it is convenient to register only with translations.  
You can do this with the following example::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha

  [STAGE]
  xform=translation
  optim=rsg
  max_its=30
  res=4 4 2

6-DOF registration (rigid)
--------------------------
The following example performs a rigid registration::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha

  [STAGE]
  xform=rigid
  optim=versor
  max_its=30
  res=4 4 2

12-DOF registration (affine)
----------------------------
The following example performs an affine registration::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=affine_coefficients.txt

  [STAGE]
  xform=affine
  optim=rsg
  max_its=30
  res=4 4 2

Demons registration
-------------------
The following example performs a demons registration::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=demons_vf.mha

  [STAGE]
  xform=vf
  optim=demons
  max_its=30
  res=4 4 2

The demons code has several parameters which can be optimized.
The following example illustrates their use::

  [STAGE]
  xform=vf
  optim=demons
  max_its=200
  res=4 4 2
  # Std dev of smoothing kernel (in mm)
  demons_std=10
  # "Gain" factor, higher gains are faster but less robust
  demons_acceleration=5
  # Homogenezation is the tradeoff between gradient 
  # and image difference.  Values should increase for larger 
  # voxel sizes, going down to about 1 for 1mm voxels.
  demons_homogenization=30
  # This is the size of the filter (in voxels)
  demons_filter_width=5 5 5

B-spline registration
---------------------
The following example performs a B-spline registration::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=bspline_coefficients.txt

  [STAGE]
  xform=bspline
  optim=lbfgsb
  # Run for at most 30 iterations
  max_its=30
  # Subsample the image 4x4x2 voxels
  res=4 4 2
  # B-spline grid spacing (in mm)
  grid_spac=30 30 30
  # Smoothness term
  regularization_lambda=0.005

Just like demons, b-spline has several options.  The most important ones 
are shown above:
res is used to subsample both input volumes prior to running the registration; 
max_its is used to determine how many iterations to run; 
grid_spac defines how far apart the control points are spaced; 
and regularization_lambda is used to increase the smoothness of the 
registration. 
The following example illustrates some additional options::

  [STAGE]
  xform=bspline
  optim=lbfgsb
  # Run for at most 30 iterations
  max_its=30
  # Subsample the image 4x4x2 voxels
  res=4 4 2
  # B-spline grid spacing (in mm)
  grid_spac=30 30 30
  # Smoothness term
  regularization_lambda=0.005
  # Quit if change in score differs by less than 3
  convergence_tol=3
  # Quit if gradient norm is less than 0.1
  grad_tol=0.1

Using ITK algorithms
--------------------
The default is to use plastimatch native implementations where available.  
When a native implementation is not available, the ITK implementation is used.
Native implementations are available for demons and bspline methods.  

If you want to use an ITK method, you can use the "impl=itk" parameter.
For example, the following command file will use the ITK demons 
implementation::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=bspline_coefficients.txt

  [STAGE]
  xform=vf
  optim=demons
  impl=itk
  max_its=30
  res=4 4 2


Mutual information
------------------
The default metric is mean squared error, which is useful for 
registration of CT with CT.  For other registration problems, mutual 
information is better.  The following example uses the Mattes 
mutual information metric with the B-spline transform::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=bspline_coefficients.txt

  [STAGE]
  xform=bspline
  impl=plastimatch
  regularization_lambda=0.005
  metric=mi
  max_its=30
  res=4 4 2

Image masking
-------------
Not all algorithms support masking.  But for those that do, you can 
specify a mask image for either the fixed image, the moving image, 
or both.  The mask image must be the same size as the input image.
When a mask image is used, only voxels of the moving or fixed image 
which have non-zero corresponding voxels in their mask image 
will contribute to the matching score.  Masks are specified as follows::

  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  fixed_mask=image_1_mask.mha
  moving_mask=image_2_mask.mha

At this time, only the plastimatch B-spline transform 
with mutual information cost function supports masking.

Output options
--------------
Outputs can be generated at the end of the registration, by putting 
the appropriate file names in the "[GLOBAL]" section.  The 
file formats of the output files are selected automatically based 
on the file extension.  

In addition to generating files at the end of registration, intermediate 
results can be generated at the end of each stage.  The following 
example shows the range of output files which can be created::

  [GLOBAL]
  # These are the inputs
  fixed=t0p_221.mha
  moving=t5p_221.mha
  xform_in=my_bsp.txt

  # These are the final outputs.  They will be rendered at full resolution.
  vf_out=my_output_vf.mha
  xform_out=my_output_bsp.txt
  img_out=my_output_img.mha

  [STAGE]
  xform=rigid
  max_its=20
  res=4 4 2

  # These are the outputs from the first stage
  xform_out=stage_1_rigid.txt
  vf_out=stage_1_rigid.mha
  img_out=stage_1_img.mha

  [STAGE]
  xform=vf
  optim=demons
  res=2 2 1

  # These are the outputs from the second stage.
  # They will be similar to the final outputs, but at lower resolution.  
  # The resolution of the stage outputs match the resolution of the stage.
  vf_out=stage_1_rigid.mha
  img_out=stage_1_img.mha

