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
       an image type string: char, uchar, short, ushort, int, uint, 
       float, or double.
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
   :widths: 20 15 10 10 45
   :header-rows: 1

   * - option
     - xform+optim+impl
     - default
     - units
     - description
   * - background_val
     - any+any+any
     - -999.0
     - image intensity
     - (needs description)
   * - convergence_tol
     - any+not demons+any
     - 5.0
     - score
     - Stop optimization if score (change?) falls below this value
   * - demons_acceleration
     - vf+demons+plastimatch
     - 1.0
     - unitless
     - (needs description)
   * - demons_filter_width
     - vf+demons+plastimatch
     - [3 3 3]
     - voxels
     - (needs description)
   * - demons_homogenization
     - vf+demons+plastimatch
     - 1.0
     - unitless
     - (needs description)
   * - demons_std
     - vf+demons+any
     - 6.0
     - mm
     - width of demons smoothing kernel
   * - histoeq
     - vf+demons+itk
     - 0
     - boolean
     - specify whether or not to equalize intensity histograms before 
       registration
   * - grad_tol
     - any+{lbfgs, lbfgsb}+any
     - 1.5
     - score per unit parameter
     - (needs description)
   * - grid_spac
     - bspline+any+any
     - [20 20 20]
     - mm
     - Spacing between control points in B-spline grid. 
       The minimum spacing is 4*(Pixel Size); if a smaller size is 
       specified, it will be adjusted upward.
   * - max_its
     - any+any+any
     - 25
     - iterations
     - (needs description)
   * - max_step
     - any+{versor, rsg}+itk
     - 10.0
     - scaled parameters
     - (needs description)
   * - metric
     - any+not demons+any
     - mse
     - string
     - Cost function metric to be optimized.  
       The choices are {mse, mi, mattes} when impl=itk, and {mse, mi} 
       when impl=plastimatch.
   * - mi_histogram_bins
     - any+any+any
     - 20
     - number of histogram bins
     - Only used for plastimatch mi metric, and itk mattes metric.
   * - min_its
     - any+any+any
     - 2
     - iterations
     - (needs description)
   * - min_step
     - any+{versor, rsg}+itk
     - 0.5
     - scaled parameters
     - (needs description)
   * - num_samples
     - any+any+itk
     - 10000
     - voxels
     - Number of voxels to randomly sample to score the cost function. 
       Only used for itk mattes metric.
   * - regularization_lambda
     - bspline+any+plastimatch
     - 0
     - unitless
     - Relative contribution of second derivative regularization 
       as compared to metric.  A typical value would range between 0.005 
       and 0.1.
   * - res
     -
     -
     -
     - Alias for "ss"
   * - ss
     - any+any+any
     - automatic
     - voxels
     - Subsampling rate for fixed and moving images.  
       This can be either "automatic", 
       a single integer (for isotropic subsampling), 
       or three integers (for anisotropic subsampling).
       In automatic mode, image is subsampled to the maximum rate 
       which yields less than 100 voxels in each dimension. 
   * - ss_fixed
     - any+any+any
     - automatic
     - voxels
     - Subsampling rate for the fixed image.
   * - ss_moving
     - any+any+any
     - automatic
     - voxels
     - Subsampling rate for the moving image.
   * - threading
     - any+any+plastimatch
     - openmp
     - string
     - Threading method used for parallel cost and gradient computations. 
       The choices are {brook, cuda, openmp, single}.  
       If an unsupported threading choice is made (such as brook with 
       b-spline), the nearest valid choice will be used.
