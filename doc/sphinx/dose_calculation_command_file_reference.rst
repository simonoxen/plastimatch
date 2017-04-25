.. _dose_calculation_command_file_reference:

Dose calculation command file reference
---------------------------------------
The dose calculation 
command file uses the "ini file" format.  

Sections
========
There are five allowed sections.  Each section 

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Section
     - Description
   * - [COMMENT]
     - Settings within a COMMENT section are ignored.
   * - [PLAN]
     - Describes global settings for the plan.  There is only a single 
       PLAN section.
   * - [BEAM]
     - Describes a single beam within a plan.  There can be multiple
       BEAM sections.
   * - [PEAK]
     - Describes a single energy or set of energies.  There can be multiple 
       PEAK sections.

PLAN options
============
The PLAN section has the following parameters

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - patient
     - Filename of CT image used for planning
   * - target
     - Filename of target volume image used for automatic 
       generation of conformal plans
   * - threading
     - Not used
   * - dose_out
     - Output filename of composite dose
   * - debug
     - ?
   * - dose_prescription
     - ?
   * - ref_dose_point
     - 3D coordinates of weight point
   * - non_normalized_dose
     - ?


BEAM options
============
The BEAM section has the following parameters

.. list-table::
   :widths: 20 80
   :header-rows: 1


   * - Option
     - Description
   * - flavor
     - ?
   * - beam_line
     - ?
   * - homo_approx
     - ?
   * - ray_step
     - ?
   * - aperture_out
     - Image of aperture, 0 means blocked by aperture, and 1 means open
       in aperture
   * - proj_dose_out
     - Image of dose in projective coordinate system of beam
   * - proj_img_out
     - Image of input patient CT in projective coordinate system of beam
   * - proj_target_out
     - Image of target in projective coordinate system of beam
   * - rc_out
     - ?
   * - particle_number_out
     - ?
   * - sigma_out
     - ?
   * - wed_out
     - ?
   * - beam_dump_out
     - Specify the directory where beam-specific debugging information
       gets saved.
   * - beam_type
     - ?
   * - beam_weight
     - ?
   * - depth_dose_z_max
     - ?
   * - depth_dose_z_res
     - ?
   * - source
     - ?
   * - isocenter
     - ?
   * - prescription_min_max
     - ?
   * - aperture_up
     - ?
   * - aperture_offset
     - ?
   * - aperture_origin
     - ?
   * - aperture_resolution
     - ?
   * - aperture_spacing
     - ?
   * - range_comp_mc_model
     - ?
   * - source_size
     - ?
   * - aperture_file_in
     - ?
   * - range_compensator_file_in
     - ?
   * - particle_number_in
     - ?
   * - aperture_smearing
     - ?
   * - proximal_margin
     - ?
   * - distal_margin
     - ?
   * - energy_resolution
     - ?
   * - energy_x
     - ?



PEAK options
============
The PEAK section has the following parameters

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - energy
     - ?
   * - spread
     - ?
   * - weight
     - ?
   * - bragg_curve
     - ?
       
