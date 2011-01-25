Programming APIs
================

Plastimatch has historically been designed as an application, and 
has only limited availability for use as an API.  However, we are trying 
to improve this situation.  The following is a work in progress toward 
making a usable API.

Registration API example
------------------------

Usage example::

  /* Load the images */
  typedef itk::Image < float, Dimension > FloatImageType;
  typedef itk::Image < FloatVectorType, Dimension > DeformationFieldType;
  FloatImageType::Pointer fixed = ...;
  FloatImageType::Pointer moving = ...;
  FloatImageType::Pointer output = ...;
  DeformationFieldType::Pointer vf = ...;
  
  /* Create the command string */
  char *command_string = 
      "[STAGE]\n"
      "xform=bspline\n"
      "max_its=30\n"
      "grid_spac=100 100 100\n"
      "res=4 4 2\n"
      ;

  /* Prepare the registration */
  Plm_registration_context *prc = plm_registration_context_create ();
  plm_registration_set_fixed (prc, &fixed);
  plm_registration_set_moving (prc, &moving);
  plm_registration_set_command_string (prc, command_string);

  /* Run the registration */
  plm_registration_execute (prc);
  if (plm_registration_get_status (prc) != 0) {
      /* Handle error */
  }

  /* Get registration outputs */
  FloatImageType::Pointer output = ...;
  DeformationFieldType::Pointer vf = ...;
  plm_registration_get_warped_image (prc, &output);
  plm_registration_get_vector_field (prc, &vf);

  /* Free the memory */
  plm_registration_context_destroy (prc);


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

