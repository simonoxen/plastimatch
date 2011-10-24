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

Image center
^^^^^^^^^^^^
The *image center* is the 2D coordinate, in pixel coordinates, of the 
point on the image which is closest to the x-ray source.  
Or to explain in another way, if you draw a line 
through the source that is also perpendicular to the image, that line 
intersects the image at the image center.  

.. image:: ../figures/proj_mat_format_1.png
   :width: 45 %

The image center is a pair of floating point numbers, in units of pixels.
The first number is the column, the second number is the row.  
The first pixel of the image is considered to be coordinate (0,0).  
The image center does not need to lie within the bounds of the image.

.. image:: ../figures/proj_mat_format_2.png
   :width: 35 %

Projection matrix
^^^^^^^^^^^^^^^^^
The *projection matrix* is the 3 x 4 matrix the maps homogenous world 
coordinates into homogenous pixel coordinates.  

A homogenous world coordinate is a 4 x 1 vector.  You can convert a 3D 
coordinate (x,y,z) into homogenous coordinates by appending a 1: (x,y,z,1).  
You can convert a homogenous coordinate (x,y,z,w) into a 3D 
coordinate by first dividing each element by w, and then taking the 
first three elements.

.. math::

   (x,y,z) \rightarrow (x,y,z,1)

.. math::

   (x,y,z,w) \rightarrow (x/w,y/w,z/w)

A similar procedure will convert 2D pixel coordinates to and 
from homogenous coordinates.

.. math::

   (i,j) \rightarrow (i,j,1)

.. math::

   (i,j,k) \rightarrow (i/k,j/k)

The projection matrix converts world coordinates into pixel coordinates.
Thus:

.. math::

   \left[\begin{array}{c} i \\ j \\ k \end{array} \right]
     =
   \left[
     \begin{array}{cccc}
     m_{11} & m_{12} & m_{13} & m_{14} \\
     m_{21} & m_{22} & m_{23} & m_{24} \\
     m_{31} & m_{32} & m_{33} & m_{34} 
     \end{array}
     \right]
   \left[\begin{array}{c} x \\ y \\ z \\ w \end{array} \right]

SID and SAD
^^^^^^^^^^^
The *SID* is the source-to-image distance, and the *SAD* is the 
source-to-axis distance.  The SID is simply the 3D distance from 
the source to the image center.  The SAD assumes a rotation axis, 
typically the axis of gantry rotation for cone-beam CT acquisition.
The units for these quantities are millimeters.

.. image:: ../figures/proj_mat_format_3.png
   :width: 35 %

Normal vector
^^^^^^^^^^^^^
The *normal vector* refers to the world coordinate normal vector of the 
imaging device.  It is the unit vector that points to the 
x-ray source as seen from the image center.

.. image:: ../figures/proj_mat_format_4.png
   :width: 35 %

Extrinsic and intrinsic matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The projection matrix is usually constructed from two components: 
the *extrinsic matrix* and the *intrinsic matrix*.  
The extrinsic matrix, C, rotates the world coordinate 
frame into a standard reference frame.
Then, the intrinsic matrix flattens out the extra dimension.  

These matrices are generated by the drr program, but aren't used 
by the fdk program.  They are included in the file format because 
they are sometimes useful for debugging.
