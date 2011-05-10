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
       an image type string, such as "float", "short", or "uchar"
       (without the quotes).
   * - background_max
     - GLOBAL
     - -1200.0 (default) Units: image intensity
       This is a threshold value that is used to automatically 
       determine the registration region of interest.

Optimization parameters.  There are three key parameters that decide
which algorithm is used for optimization. 

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - xform
     - optim
     - impl
   * - align_center
     - N/A
     - itk
   * - translation
     - rsg, amoeba
     - itk
   * - rigid
     - versor, amoeba
     - itk
   * - affine
     - rsg, amoeba
     - itk
   * - vf
     - demons
     - plastimatch, itk
   * - bspline
     - steepest, lbfgs, lbfgsb
     - plastimatch, itk

Notes:

#. Default values are: xform=rigid, optim=versor, impl=plastimatch.
#. Amoeba is reported not to work well.
#. B-spline with steepest descent optimization is only supported on
   plastimatch implementation.
#. B-spline with lbfgs optimization is only supported on itk implementation.

The following specific parameters are used to refine the optimization.
Depending on the choice of xform, optim, and impl, a different set of
specific parameters are available. 

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - option
     - xform+optim+impl
     - default
     - description
   * - res
     - any+any+any
     - automatic
     - [1 1 1] (minimum) Units: voxels, must be integers. 
       In automatic mode, image is subsampled to less than 100 
       voxels in each dimension. 
   * - metric
     - any+not demons+any
     - mse
     - Choices are: {mse, mi, mattes} when impl=itk, and {mse, mi} 
       when impl=plastimatch.
   * - background_val
     - any+any+any
     - -999.0
     - Units: image intensity
   * - min_its
     - any+any+any
     - 2
     - Units: iterations
   * - max_its
     - any+any+any
     - 25
     - Units: iterations
   * - convergence_tol
     - any+not demons+any
     - 5.0
     - Units: score
   * - grad_tol
     - any+{lbfgs, lbfgsb}+any
     - 1.5
     - Units: score per unit parameter
   * - min_step
     - any+{versor, rsg}+itk
     - 0.5
     - Units: scaled parameters
   * - max_step
     - any+{versor, rsg}+itk
     - 10.0
     - Units: scaled parameters
   * - mi_histogram_bins
     - any+any+any
     - 20
     - Number of histogram bins.  Only used for plastimatch mi metric, and 
       itk mattes metric.
   * - mi_num_spatial_samples
     - any+any+itk
     - 10000
     - Number of spatial samples.  Only used for itk mattes metric.
   * - grid_spac
     - bspline+any+any
     - [20 20 20]
     - Units: mm. Minimum size is 4*(Pixel Size).  If a smaller size is 
       specified, it will be adjusted upward.
   * - histoeq
     - vf+demons+itk
     - 0
     - Specifies whether or not to equalize intensity histograms before 
       registration.
   * - demons_std
     - vf+demons+any
     - 6.0
     - Units: mm.  Width of demons smoothing kernel.
   * - demons_acceleration
     - vf+demons+plastimatch
     - 1.0
     - Unitless.
   * - demons_homogenization
     - vf+demons+plastimatch
     - 1.0
     - Unitless.
   * - demons_filter_width
     - vf+demons+plastimatch
     - [3 3 3]
     - Units: voxels.
