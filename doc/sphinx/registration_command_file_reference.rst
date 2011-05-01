.. _registration_command_file_reference:

Registration command file reference
-----------------------------------

The parameter file has two sections: a GLOBAL section at the top of
the file, and one or more STAGE section. Parameters such as input
files are put only in the GLOBAL section. Output files can be put in
the GLOBAL section or any STAGE section (which will write out
intermediate output).

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - option
     - stage
     - value
   * - fixed
     - GLOBAL
     - Filename of fixed image
   * - moving
     - GLOBAL
     - Filename of moving image
   * - xform_in
     - GLOBAL, STAGE
     - Initial guess for transform
   * - xform_out
     - GLOBAL, STAGE
     - Filename of output transform
   * - vf_out
     - GLOBAL, STAGE
     - Filename of output transform, as vector field
   * - img_out
     - GLOBAL, STAGE
     - Filename of warped image
   * - img_out_fmt
     - GLOBAL, STAGE
     - Output format, which must be either “auto” (default), 
       which means the filename extenstion is used to determine
       the file format, or “dicom”, which interprets img_out 
       as a directory name to output the dicom files
   * - img_out_type
     - GLOBAL, STAGE
     - Data type of the output image.  Either “auto” (default), or 
       and image type string, such as "float", "short", or "uchar"
       (without the quotes).
   * - background_max
     - GLOBAL
     - -1200.0 (default) Units: image intensity
       This is a threshold value that is used to automatically 
       determine the registration region of interest.

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


+----------------------+----------------+------------+---------------------------+
|option                |xform+optim+impl|default     |description                |
|                      |                |            |                           |
|                      |                |            |                           |
+======================+================+============+===========================+
|res                   |any+any+any     |Automatic   |[1 1 1] (minimum) Units:   |
|                      |                |            |voxels, must be            |
|                      |                |            |integers. In automatic     |
|                      |                |            |mode, image is subsampled  |
|                      |                |            |to less than 100 voxels in |
|                      |                |            |each dimension.            |
+----------------------+----------------+------------+---------------------------+
|metric                |any+not         |mse         |Choices are: {mse, mi,     |
|                      |demons+any      |            |mattes} when impl=itk,     |
|                      |                |            |{mse, mi} when             |
|                      |                |            |impl=plastimatch           |
+----------------------+----------------+------------+---------------------------+
|background_val        |any+any+any     |-999.0      |Units: image intensity     |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|min_its               |any+any+any     |2           |Units: iterations          |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|max_its               |any+any+any     |25          |Units: iterations          |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|convergence_tol       |any+not         |5.0         |Units: score               |
|                      |demons+any      |            |                           |
|                      |                |            |                           |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|grad_tol              |any+{lbfgsb or  |1.5         |Units: score per unit      |
|                      |lbfgs}+any      |            |parameter                  |
+----------------------+----------------+------------+---------------------------+
|max_step              |any+{versor or  |10.0        |Units: scaled parameters   |
|                      |rsg}+itk        |            |                           |
+----------------------+----------------+------------+---------------------------+
|min_step              |any+{versor or  |0.5         |Units: scaled parameters   |
|                      |rsg}+itk        |            |                           |
+----------------------+----------------+------------+---------------------------+
|mi_histogram_bins     |any+any+any     |20          |Number of histogram        |
|                      |                |            |bins. Only for used for    |
|                      |                |            |plastimatch mi or itk      |
|                      |                |            |mattes metrics             |
+----------------------+----------------+------------+---------------------------+
|mi_num_spatial_samples|any+any+itk     |10000       |Number of spatial          |
|                      |                |            |samples.  Only for itk     |
|                      |                |            |mattes metric              |
+----------------------+----------------+------------+---------------------------+
|grid_spac             |bspline+any+any |[20 20 20]  |Units: mm. Minimum size    |
|                      |                |            |is 4*(Pixel Size).  If a   |
|                      |                |            |smaller size is            |
|                      |                |            |specified, it will be      |
|                      |                |            |adjusted upward.           |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|histoeq               |vf+demons+itk   |0           |Specifies whether or not   |
|                      |                |            |to equalize intensity      |
|                      |                |            |histograms before          |
|                      |                |            |registration.              |
+----------------------+----------------+------------+---------------------------+
|demons_std            |vf+demons+any   |6.0         |Units: mm                  |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|demons_acceleration   |vf+demons +     |1.0         |Units: percent             |
|                      |plastimatch     |            |                           |
+----------------------+----------------+------------+---------------------------+
|demons_homogenization |vf+demons +     |1.0         |Untiless                   |
|                      |plastimatch     |            |                           |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
|demons_filter_width   |vf+demons +     |[3 3 3]     |Units: voxels.             |
|                      |plastimatch     |            |                           |
|                      |                |            |                           |
+----------------------+----------------+------------+---------------------------+
