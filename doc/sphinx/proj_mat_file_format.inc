.. _proj_mat_file_format:

Projection matrix file format
-----------------------------

The projection matrices are stored in an ASCII file format, and 
include the complete geometry needed for DRR computation 
or CBCT reconstruction.  These files are created by the program 
"drr", and used by the program "fdk".

The following example is a valid projection matrix file::

    6.35000000e+01     6.35000000e+01
    0.00000000e+00     2.13333333e-01     0.00000000e+00     0.00000000e+00
    0.00000000e+00     0.00000000e+00    -2.13333333e-01     0.00000000e+00
   -6.13496933e-04     0.00000000e+00     0.00000000e+00     6.13496933e-01
    1.00000000e+03
    1.63000000e+03
   -1.00000000e+00    -0.00000000e+00    -0.00000000e+00
 Extrinsic
   -0.00000000e+00     1.00000000e+00    -0.00000000e+00     0.00000000e+00
    0.00000000e+00    -0.00000000e+00    -1.00000000e+00     0.00000000e+00
   -1.00000000e+00    -0.00000000e+00    -0.00000000e+00     1.00000000e+03
    0.00000000e+00     0.00000000e+00     0.00000000e+00     1.00000000e+00
 Intrinsic
    2.13333333e-01     0.00000000e+00     0.00000000e+00     0.00000000e+00
    0.00000000e+00     2.13333333e-01     0.00000000e+00     0.00000000e+00
    0.00000000e+00     0.00000000e+00     6.13496933e-04     0.00000000e+00

The meaning of each of the fields is explained below.  Note, however, that 
comments are not allowed, so this example cannot be loaded 
without first removing the explanations::

    # Image center (in pixels)
    6.35000000e+01     6.35000000e+01
    # Projection matrix
    0.00000000e+00     2.13333333e-01     0.00000000e+00     0.00000000e+00
    0.00000000e+00     0.00000000e+00    -2.13333333e-01     0.00000000e+00
   -6.13496933e-04     0.00000000e+00     0.00000000e+00     6.13496933e-01
    # SAD
    1.00000000e+03
    # SID
    1.63000000e+03
    # Normal vector
   -1.00000000e+00    -0.00000000e+00    -0.00000000e+00
    # Extrinsic portion of projection matrix
 Extrinsic
   -0.00000000e+00     1.00000000e+00    -0.00000000e+00     0.00000000e+00
    0.00000000e+00    -0.00000000e+00    -1.00000000e+00     0.00000000e+00
   -1.00000000e+00    -0.00000000e+00    -0.00000000e+00     1.00000000e+03
    0.00000000e+00     0.00000000e+00     0.00000000e+00     1.00000000e+00
    # Intrinsic portion of projection matrix
 Intrinsic
    2.13333333e-01     0.00000000e+00     0.00000000e+00     0.00000000e+00
    0.00000000e+00     2.13333333e-01     0.00000000e+00     0.00000000e+00
    0.00000000e+00     0.00000000e+00     6.13496933e-04     0.00000000e+00

