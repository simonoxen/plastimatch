Image registration
==================

Quick start guide
-----------------
You must create a command file to do registration.  
If you want to register image_2.mha to match image_1.mha using 
B-spline registration, create a command file like this::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=bspline_coefficients.txt

  [STAGE]
  xform=bspline
  impl=plastimatch
  threading=openmp
  max_its=30
  grid_spac=100 100 100
  res=4 2 2

Then, run the registration like this::

  plastimatch register command_file.txt

The above example only performs a single registration stage.  If you 
want to do multi-stage registration, use multiple [STAGE] sections.  
Like this::

  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=bspline_coefficients.txt

  [STAGE]
  xform=bspline
  impl=plastimatch
  threading=openmp
  max_its=30
  grid_spac=100 100 100
  res=4 2 2

  [STAGE]
  max_its=30
  grid_spac=80 80 80
  res=2 2 1

  [STAGE]
  max_its=30
  grid_spac=60 60 60
  res=1 1 1

That concludes the quick start guide.  For more details and 
examples, read on!

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
  res=4 2 2

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
  res=4 2 2

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
  res=4 2 2

Using ITK algorithms
--------------------
(To be written)


Mutual information
------------------
The default metric is mean squared error, which is useful for 
registration of CT with CT.  For other registration problems, mutual 
information is better.  The first example uses the Mattes 
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
  metric=mi
  max_its=30
  res=4 2 2



  # command_file.txt
  [GLOBAL]
  fixed=image_1.mha
  moving=image_2.mha
  img_out=warped_2.mha
  xform_out=bspline_coefficients.txt

  [STAGE]
  xform=bspline
  optim=lbfgsb
  metric=mi
  max_its=30
  res=4 2 2


The command file is an ASCII text file, which controls the registration. 
An example registration parameter file is given as follows::

	## Comments begin with '#', blank lines are ignored
	## A "GLOBAL" section is required, and specifies the inputs 
	## and optional outputs.  
	## 
	## For the output files, it is similar, but not exactly the same 
	## as putting them in the last stage.  
	## 
	## The use of xform_out is undefined when xform=vf, 
	## such as demons registration.  Use vf_out instead.
	[GLOBAL]
	fixed=t0p_221.mha
	moving=t5p_221.mha
	xform_in=my_bsp.txt
	vf_out=my_output_vf.mha
	xform_out=my_output_bsp.txt
	img_out=my_output_img.mha

	## Use a "STAGE" section for each stage of processing.  
	## You can get outputs (vf_out, xform_out, and img_out) if 
	## you like. 
	## Default values will be used for each option unless you 
	## override them here. 
	[STAGE]
	xform=bspline
	optim=lbfgsb
	# Use plastimatch native implementation
	impl=plastimatch
	# Quit after 50 iterations
	max_its=50
	# Quit if change in score differs by less than 3
	convergence_tol=3
	# Quit if gradient norm is less than 0.1
	grad_tol=0.1
	# B-spline grid spacing (in mm)
	grid_spac=30 30 30
	# Subsample both files by 4,4,2 in x,y,z directions
	res=4 4 2
	# At the end of this stage, output the final xform
	xform_out=stage_1_output_bsp.txt

	## For each subsequent stage, add another "STAGE" section.
	## You only need to include parameters that you want to change.
	## Any unspecified parameter is inherited from the 
	## previous stage.
	[STAGE]
	max_its=20
	convergence_tol=4
	grad_tol=0.2
	res=2 2 1

	## To run demons, set optim to "demons" and xform to "vf." 
	[STAGE]
	xform=vf
	optim=demons
	impl=itk
	vf_out=output_2.mha

	## If you want to, you can use a "COMMENT" section 
	## to comment out a stage.
	[COMMENT]
	xform=bspline
	max_its=200

	## GPU acceleration requires brook
	[STAGE]
	optim=demons
	xform=vf
	impl=gpuit_brook
	res=4 4 2
	max_its=200
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

Registration command file reference
-----------------------------------

The parameter file has two sections: a GLOBAL section at the top of
the file, and one or more STAGE section. Parameters such as input
files are put only in the GLOBAL section. Output files can be put in
the GLOBAL section or any STAGE section (which will write out
intermediate output).

+--------------+-------+-------------------------------------------+
|option        |stage  |value                                      |
+==============+=======+===========================================+
|fixed         |GLOBAL |Name of fixed image                        |
|              |       |                                           |
+--------------+-------+-------------------------------------------+
|moving        |GLOBAL |Name of fixed image                        |
|              |       |                                           |
+--------------+-------+-------------------------------------------+
|xform_in      |GLOBAL,|Initial guess                              |
|              |STAGE  |                                           |
+--------------+-------+-------------------------------------------+
|xform_out     |GLOBAL,|Final transform                            |
|              |STAGE  |                                           |
+--------------+-------+-------------------------------------------+
|vf_out        |GLOBAL,|Final transform, as vector field           |
|              |STAGE  |                                           |
+--------------+-------+-------------------------------------------+
|img_out       |GLOBAL,|Warped image                               |
|              |STAGE  |                                           |
+--------------+-------+-------------------------------------------+
|img_out_fmt   |GLOBAL,|“auto” (default) Output format Must be     |
|              |STAGE  |either “auto”, which uses filename         |
|              |       |extenstion to determine, or “dicom”, which |
|              |       |iterprets img_out as a directory name to   |
|              |       |output the dicom files                     |
|              |       |                                           |
|              |       |                                           |
+--------------+-------+-------------------------------------------+
|img_out_type  |GLOBAL,|“auto” (default) Data type of the output   |
|              |STAGE  |image, usually either float, short, or     |
|              |       |uchar                                      |
|              |       |                                           |
+--------------+-------+-------------------------------------------+
|background_max|GLOBAL |-1200.0 (default) Units: image intensity   |
|              |       |This is used to automatically determine a  |
|              |       |region of interest                         |
|              |       |                                           |
|              |       |                                           |
+--------------+-------+-------------------------------------------+

Optimization parameters.  There are three key parameters that decide
which algorithm is used for optimization. 

+--------------+---------+-------------------------------------------+
|xform         |optim    |impl                                       |
+==============+=========+===========================================+
|align_center  |N/A      |itk                                        |
|              |         |                                           |
+--------------+---------+-------------------------------------------+
|translation   |rsg,     |itk                                        |
|              |amoeba   |                                           |
+--------------+---------+-------------------------------------------+
|rigid         |versor,  |itk                                        |
|              |amoeba   |                                           |
+--------------+---------+-------------------------------------------+
|affine        |rsg,     |itk                                        |
|              |amoeba   |                                           |
+--------------+---------+-------------------------------------------+
|vf            |demons   |plastimatch, itk                           |
+--------------+---------+-------------------------------------------+
|bspline       |steepest,|plastimatch, itk                           |
|              |lbfgs,   |                                           |
|              |lbfgsb   |                                           |
+--------------+---------+-------------------------------------------+

Notes:

#. Default values are: xform=rigid, optim=versor, impl=plastimatch.
#. Amoeba is reported not to work well.
#. B-spline with steepest descent optimization is only supported on
   plastimatch implementation.
#. B-spline with lbfgs optimization is only supported on itk implementation.

The following specific parameters are used to refine the optimization.
Depending on the choice of xform, optim, and impl, a different set of
specific parameters are available. 

+----------------------+----------------+------------+-------------------------+
|option                |xform+optim+impl|default     |description              |
+======================+================+============+=========================+
|res                   |any+any+any     |[4 4 1]     |[1 1 1] (minimum) Units: |
|                      |                |            |voxels Must be integers  |
|                      |                |            |                         |
|                      |                |            |                         |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|metric                |any+not         |mse         |mse, mi, mattes          |
|                      |demons+any      |            |(impl=itk) mse, mi       |
|                      |                |            |(impl=gpuit_cpu)         |
+----------------------+----------------+------------+-------------------------+
|background_val        |any+any+any     |-999.0      |Units: image intensity   |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|min_its               |any+any+any     |2           |Units: iterations        |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|max_its               |any+any+any     |25          |Units: iterations        |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|convergence_tol       |any+not         |5.0         |Units: score             |
|                      |demons+any      |            |                         |
|                      |                |            |                         |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|grad_tol              |any+{lbfgsb or  |1.5         |Units: score per unit    |
|                      |lbfgs}+any      |            |parameter                |
+----------------------+----------------+------------+-------------------------+
|max_step              |any+{versor or  |10.0        |Units: scaled parameters |
|                      |rsg}+itk        |            |                         |
+----------------------+----------------+------------+-------------------------+
|min_step              |any+{versor or  |0.5         |Units: scaled parameters |
|                      |rsg}+itk        |            |                         |
+----------------------+----------------+------------+-------------------------+
|mi_histogram_bins     |any+any+any     |20          |Number of histogram      |
|                      |                |            |bins. Only for used for  |
|                      |                |            |plastimatch mi or itk    |
|                      |                |            |mattes metrics           |
+----------------------+----------------+------------+-------------------------+
|mi_num_spatial_samples|any+any+itk     |10000       |Number of spatial        |
|                      |                |            |samples.  Only for itk   |
|                      |                |            |mattes metric            |
+----------------------+----------------+------------+-------------------------+
|grid_spac             |bspline+any+any |[20 20 20]  |Units: mm. Minimum size  |
|                      |                |            |is 4*(Pixel Size).  If a |
|                      |                |            |smaller size is          |
|                      |                |            |specified, it will be    |
|                      |                |            |adjusted upward.         |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|histoeq               |vf+demons+itk   |0           |Specifies whether or not |
|                      |                |            |to equalize intensity    |
|                      |                |            |histograms before        |
|                      |                |            |registration.            |
+----------------------+----------------+------------+-------------------------+
|demons_std            |vf+demons+any   |6.0         |Units: mm                |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|demons_acceleration   |vf+demons +     |1.0         |Units: percent           |
|                      |plastimatch     |            |                         |
+----------------------+----------------+------------+-------------------------+
|demons_homogenization |vf+demons +     |1.0         |Untiless                 |
|                      |plastimatch     |            |                         |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+
|demons_filter_width   |vf+demons +     |[3 3 3]     |Units: voxels.           |
|                      |plastimatch     |            |                         |
|                      |                |            |                         |
+----------------------+----------------+------------+-------------------------+

