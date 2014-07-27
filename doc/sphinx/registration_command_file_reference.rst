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
   * - fixed_mask
     - GLOBAL
     - Filename of a binary mask for the fixed image; 
       only pixels which are non-zero in this image will contribute 
       to the registration result
   * - moving_mask
     - GLOBAL
     - Filename of a binary mask for the moving image;
       only pixels which are non-zero in this image will contribute 
       to the registration result
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
which algorithm is used for optimization. One additional parameter( optim_subtype) is
only available for ITK demons and determines which ITK demons algorithm is used: 

.. list-table::
   :widths: 20 40 40 40
   :header-rows: 1

   * - xform
     - optim
     - impl
     - optim_subtype
   * - align_center
     - N/A
     - itk
     - ---
   * - translation
     - rsg, amoeba
     - itk
     - ---
   * - rigid
     - versor, amoeba
     - itk
     - ---
   * - affine
     - rsg, amoeba
     - itk
     - ---
   * - vf
     - demons
     - plastimatch, itk
     - fsf(default),diffeomorphic,log_domain,sym_log_domain (only for impl=ITK) 
   * - bspline
     - steepest, lbfgs, lbfgsb
     - plastimatch, itk
     - ---

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
     - 1e-6
     - score
     - Stop optimization if change in score between iterations 
       falls below this value
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
   * - demons_gradient_type
     - vf+demons+itk
     - symmetric
     - enumeration
     - Type of gradient that will be used to compute update force, choose from {symmetric,fixed,warped_moving,mapped_moving}
   * - demons_smooth_update_field
     - vf+demons+itk
     - false
     - bool
     - Set whether the update field is smoothed
   * - demons_std_update_field
     - vf+demons+itk
     - 1
     - std-dev.
     - Width of Gaussian used to smooth update field
   * - demons_smooth_deformation_field
     - vf+demons+itk
     - true
     - bool
     - Set whether the deformation field is smoothed
   * - demons_std_deformation_field
     - vf+demons+itk
     - 1
     - std-dev.
     - Width of Gaussian used to smooth deformation field
   * - demons_step_length
     - vf+demons+itk
     - 1
     - mm
     - maximum update step length. 0 implies no restriction 
   * - histoeq
     - vf+demons+itk
     - 0
     - boolean
     - specify whether or not to equalize intensity histograms before 
       registration
   * - num_hist_levels_equal
     - vf+demons+itk
     - 1000
     - unsigned int
     - set number of histogram levels for histogram equalization
   * - num_matching_points
     - vf+demons+itk
     - 500
     - unsigned int
     - set number of histogram levels for histogram equalization
   * - thresh_mean_intensity
     - vf+demons+itk
     - 0
     - boolean
     - Set the threshold at mean intensity flag. If true, only source (reference) pixels which are greater than the mean source (reference) intensity is used in the histogram matching. If false, all pixels are used.
   * - grad_tol
     - any+{lbfgs}+itk
     - 1.5
     - score per unit parameter
     - Gradient convergence tolerance for LBFGS optimizer.
       The optimizer can be asked to stop when the gradient
       magnitude is below this number.
   * - pgtol
     - any+{lbfgsb}+any
     - 1e-5
     - score per unit parameter
     - Projected gradient tolerance for LBFGSB optimizer.
       The optimizer can be asked to stop when the projected gradient
       is below this number.  The projected gradient is defined 
       as max{proj g_i | i = 1, ..., n} 
       where proj g_i is the ith component of the projected gradient.
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
   * - mattes_fixed_minVal, mattes_fixed_maxVal
     - bspline+any+itk
     - 0
     - image intensity
     - Min and max intensity values of intensity range for fixed image used for MI calculation.
       If values are not set by user min and max values will be calculated from images. Only used for optimized version of itk implementation.
   * - mattes_moving_minVal, mattes_moving_maxVal
     - bspline+any+itk
     - 0
     - image intensity
     - Min and max intensity values of intensity range for moving image used for MI calculation.
       If values are not set by user min and max values will be calculated from images. Only used for optimized version of itk implementation.
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
     - -1
     - voxels
     - Number of voxels to randomly sample to score the cost function. 
       Only used for itk mattes metric.  If this parameter is not 
       specified, num_samples_pct will be used instead.
   * - num_samples_pct
     - any+any+itk
     - 0.3
     - percent
     - Percent of voxels to randomly sample to score the cost function. 
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
   * - res_vox
     - any+any+any
     - automatic
     - voxels
     - Subsampling rate for fixed and moving images.  
       This can be either "automatic", 
       a single integer (for isotropic subsampling), 
       or three integers (for anisotropic subsampling).
       In automatic mode, image is subsampled to the maximum rate 
       which yields less than 100 voxels in each dimension. 
   * - rsg_grad_tol
     - any+{rsg, versor}+itk
     - 0.001
     - score per unit parameter
     - Gradient magnitude tolerance for RSG and Versor optimizers.
       The optimizer can be asked to stop when the cost function is 
       in a stable region where the gradient magnitude is smaller 
       than this value.
   * - ss
     -
     -
     -
     - Alias for "res_vox"
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
       The choices are {cuda, opencl, openmp, single}.  
       If an unsupported threading choice is made (such as cuda with 
       demons), the nearest valid choice will be used.
