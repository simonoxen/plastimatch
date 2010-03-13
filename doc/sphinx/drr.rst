DRR - Digitally reconstructed radiographs
=========================================


DRR Geometry
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

