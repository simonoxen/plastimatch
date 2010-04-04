DRR - Digitally reconstructed radiographs
=========================================

A digitally reconstructed radiograph (DRR) is a synthetic radiograph 
which can be generated from a computed tomography (CT) scan.  
It is used as a reference image for verifying the correct setup 
position of a patient prior to radiation treatment.  

DRR usage
---------
The drr program that comes with plastimatch takes a CT image 
as input, and generates one or more output images.  The input image 
is in MHA format, and the output images can be either pgm, pfm, or raw 
format.  The command line usage is::

 Usage: drr [options] [infile]
 Options:
   -A hardware       Either "cpu" or "brook" or "cuda" (default=cpu)
   -a num            Generate num equally spaced angles
   -N ang            Difference between neighboring angles (in degrees)
   -nrm "x y z"      Set the normal vector for the panel
   -vup "x y z"      Set the vup vector (toward top row) for the panel
   -g "sad sid"      Set the sad, sid (in mm)
   -r "r1 r2"        Set output resolution (in pixels)
   -s scale          Scale the intensity of the output file
   -e                Do exponential mapping of output values
   -c "c1 c2"        Set the image center (in pixels)
   -z "s1 s2"        Set the physical size of imager (in mm)
   -w "w1 w2 w3 w4"  Only produce image for pixes in window (in pix)
   -t outformat      Select output format: pgm, pfm or raw
   -S                Output multispectral output files
   -i exact          Use exact trilinear interpolation
   -i approx         Use approximate trilinear interpolation
   -o "o1 o2 o3"     Set isocenter position
   -I infile         Set the input file in mha format
   -O outprefix      Generate output files using the specified prefix

Single image mode
-----------------
The drr program can be used in either 
*single image mode* or *rotational mode*.  In single image mode, 
you must specify the complete geometry of the x-ray source and imaging 
panel.  
The following example illustrates a complete geometry specification::

  drr -nrm "1 0 0" \
      -vup "0 0 1" \
      -g "1000 1500" \
      -r "768 1024" \
      -z "300 400" \
      -c "383.5 511.5" \
      -o "0 -20 -50" \
      input_file.mha

In the above example, the isocenter is chosen to be 
(0, -20, -50), the location marked on the 
CT image.  The orientation of the projection image is controlled by 
the **nrm** and **vup** options.  Using the default values of (1, 0, 0) 
and (0, 0, 1) yields the DRR shown on the right:

.. image:: ../figures/drr_input.png
   :width: 37 %

.. image:: ../figures/drr_output_1.png
   :width: 50 %

By changing the normal direction (**nrm**), we can choose different 
beam direction within an isocentric orbit.  For example, an 
anterior-posterior (AP) DRR is generated with a normal of (0, -1, 0) 
as shown below:

.. image:: ../figures/drr_output_2.png
   :width: 50 %


Rotational mode
---------------
(Add documentation here)

DRR geometry
------------

The DRR code generates images from a volume, using the following
geometry description:

+-----------+-----------------+-----------------------------------------+
|Inputs     |Units            |Default Value                            |
|           |                 |                                         |
+===========+=================+=========================================+
|SID        |mm               |1630                                     |
+-----------+-----------------+-----------------------------------------+
|SAD        |mm               |1000                                     |
+-----------+-----------------+-----------------------------------------+
|Angle      |degrees          |0                                        |
+-----------+-----------------+-----------------------------------------+
|Target     |3d position (mm) |(0,0,0)                                  |
|           |                 |                                         |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|Camera     |3d position (mm) |*Computed from Target, Angle and SAD*    |
|           |                 |                                         |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|CT Zero    |3d position (vox)|*Read from input volume*                 |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|Panel      |pixels           |1024 x 768                               |
|resolution |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|Panel size |mm               |400 x 300                                |
+-----------+-----------------+-----------------------------------------+
|Window     |pixels           |Same as resolution                       |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|NRM        |3d direction     |*Computed from Target and Camera*        |
|           |                 |                                         |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|VUP        |3d direction     |*Hard coded to (0,0,1)*                  |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+
|Panel      |3d direction     |*Computed from NRM and VUP*              |
|Orientation|                 |                                         |
|           |                 |                                         |
+-----------+-----------------+-----------------------------------------+


.. figure:: ../figures/drr_geometry.png
   :width: 80 %

   Geometry attributes of a DRR

.. figure:: ../figures/drr_intrinsic.png
   :width: 50 %

   Intrinsic geometry for DRR computation

The intrinsic geometry is specified by the equation:

.. math::

   K = \left[
     \begin{array}{cccc}
     1/\alpha & 0 & 0 & c_i \\
     0 & 1 / \beta & 0 & c_j \\
     0 & 0 & f & 0
     \end{array}
     \right]


DRR API example
---------------

Usage example::

  Volume *vol;
  Proj_image *proj;
  
  /* Create the CT volume */
  int dim[3] = { 512, 512, 100 };
  float offset[3] = { -255.5, -255.5, -123.75 };
  float spacing[3] = { 1.0, 1.0, 2.5 };
  enum Volume_pixel_type pix_type = PT_FLOAT;
  float direction_cosines = { 
  	1.0, 0.0, 0.0,
  	0.0, 1.0, 0.0,
  	0.0, 0.0, 1.0 };
  vol = volume_create (dim, offset, spacing, pix_type, direction_cosines, 0);
  
  /* Fill in the CT volume with values */
  float *img = (float*) vol->img;
  img[100] = 32.6;
  
  /* Create empty projection image */
  proj = proj_image_create ();
  /* Add storage for image bytes */
  proj_image_create_img (proj, ires);
  /* Add empty projection matrix */
  proj_image_create_pmat (proj);
  
  /* Set up the projection matrix */
  proj_matrix_set (proj->pmat, cam, tgt, vup, sid, ic, ps, ires);
  
  /* Render the drr */
  drr_render_volume_perspective (proj, vol, ps, 0, options);
  
  /* Do something with the image */
  printf (“pixel (32,10) is: %g\n”, proj->img[32*ires[0]+10]);
  
  /* Clean up memory */
  volume_destroy (vol);
  proj_image_destroy (proj);

