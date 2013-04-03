/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "aperture.h"
#include "plm_image.h"
#include "plm_math.h"
#include "proj_volume.h"
#include "proton_beam.h"
#include "proton_scene.h"
#include "ray_trace_probe.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_limit.h"
#include "wed_parms.h"

typedef struct callback_data Callback_data;
struct callback_data {
    Volume* wed_vol;   /* Water equiv depth volume */
    int* ires;         /* Aperture Dimensions */
    int ap_idx;        /* Current Aperture Coord */
};


static int
skin_ct (Volume* ct_volume, Volume* skin_volume, float background)
{
  float *ct_img = (float*) ct_volume->img;
  float *skin_img = (float*) skin_volume->img;

  const plm_long *ct_dim = ct_volume->dim; 
  const plm_long *skin_dim = skin_volume->dim;

  if ((ct_dim[0]!=skin_dim[0])||(ct_dim[1]!=skin_dim[1])||(ct_dim[2]!=skin_dim[2]))  {
        fprintf (stderr, "\n** ERROR: CT dimensions do not match skin dimensions.\n");
	return -1;
  }

  plm_long n_voxels = ct_dim[0]*ct_dim[1]*ct_dim[2];
  
  for (plm_long i=0; i!=n_voxels; ++i)  {

    if (skin_img[i] == 0)  {ct_img[i] = background;}
    else if (skin_img[i] == 1)  {continue;}
    else {
      fprintf (stderr, "\n** ERROR: Value other than '0' or '1' in skin input.\n");
      return -1;
    }
  }

  return 0;
}

static Volume*
create_wed_volume (Wed_Parms* parms, Proton_Scene *scene)
{
    Rpl_volume* rpl_vol = scene->rpl_vol;

    float wed_off[3] = {0.0f, 0.0f, 0.0f};
    float wed_ps[3] = {1.0f, 1.0f, 1.0f};

    /* water equivalent depth volume has the same x,y dimensions as the rpl
     * volume. Note: this means the wed x,y dimensions are equal to the
     * aperture dimensions and the z-dimension is equal to the sampling
     * resolution chosen for the rpl */
    plm_long wed_dims[3];

    Volume *vol = rpl_vol->get_volume ();
    wed_dims[0] = vol->dim[0];
    wed_dims[1] = vol->dim[1];
    wed_dims[2] = vol->dim[2];


    return new Volume (wed_dims, wed_off, wed_ps, NULL, PT_FLOAT, 1);
}

static Volume*
create_dew_volume (Wed_Parms* parms, Proton_Scene *scene)
{
 
    Volume* patient_vol = scene->get_patient_vol();

    float dew_off[3];
    dew_off[0] = patient_vol->offset[0];
    dew_off[1] = patient_vol->offset[1];
    dew_off[2] = patient_vol->offset[2];

    float dew_ps[3];
    dew_ps[0] = patient_vol->spacing[0];
    dew_ps[1] = patient_vol->spacing[1];
    dew_ps[2] = patient_vol->spacing[2];

    plm_long dew_dims[3];
    dew_dims[0] = patient_vol->dim[0];
    dew_dims[1] = patient_vol->dim[1];
    dew_dims[2] = patient_vol->dim[2];

    //If output volume dimensions were set in .cfg file, use these.
    if (parms->dew_dim[0]!=-999.) {dew_dims[0]=parms->dew_dim[0];}
    if (parms->dew_dim[1]!=-999.) {dew_dims[1]=parms->dew_dim[1];}
    if (parms->dew_dim[2]!=-999.) {dew_dims[2]=parms->dew_dim[2];}

    if (parms->dew_origin[0]!=-999.) {dew_off[0]=parms->dew_origin[0];}
    if (parms->dew_origin[1]!=-999.) {dew_off[1]=parms->dew_origin[1];}
    if (parms->dew_origin[2]!=-999.) {dew_off[2]=parms->dew_origin[2];}

    if (parms->dew_spacing[0]!=-999.) {dew_ps[0]=parms->dew_spacing[0];}
    if (parms->dew_spacing[1]!=-999.) {dew_ps[1]=parms->dew_spacing[1];}
    if (parms->dew_spacing[2]!=-999.) {dew_ps[2]=parms->dew_spacing[2];}

    return new Volume (dew_dims, dew_off, dew_ps, NULL, PT_FLOAT, 1);
}

static Volume*
create_segdepth_volume (Wed_Parms* parms, Proton_Scene *scene)
{
 
    Rpl_volume* rpl_vol = scene->rpl_vol;

    /* water equivalent depth volume has the same x,y dimensions as the rpl
     * volume. Note: this means the wed x,y dimensions are equal to the
     * aperture dimensions and the z-dimension is equal to the sampling
     * resolution chosen for the rpl */
    plm_long segdepth_dims[3];

    Volume *vol = rpl_vol->get_volume ();
    segdepth_dims[0] = vol->dim[0];
    segdepth_dims[1] = vol->dim[1];
    segdepth_dims[2] = 1;

  
    float segdepth_off[3];
    segdepth_off[0] = 0;
    segdepth_off[1] = 0;
    segdepth_off[2] = 0;

    float segdepth_ps[3];
    segdepth_ps[0] = parms->ap_spacing[0];
    segdepth_ps[1] = parms->ap_spacing[1];
    segdepth_ps[2] = 1;

    return new Volume (segdepth_dims, segdepth_off, segdepth_ps, NULL, PT_FLOAT, 1);
}

void
wed_ct_compute (
    const char* out_fn,
    Wed_Parms* parms,
    Plm_image *ct_vol,
    Proton_Scene *scene,
    float background
)
{

  Rpl_volume* rpl_vol = scene->rpl_vol;

    if (parms->mode==0)  {
      Volume* wed_vol;
      
      wed_vol = create_wed_volume (parms, scene);
      rpl_vol->compute_wed_volume (wed_vol, ct_vol->gpuit_float(), background);
      plm_image_save_vol (out_fn, wed_vol);
    }

    if (parms->mode==1)  {
      Volume* dew_vol;
      
      dew_vol = create_dew_volume (parms, scene);
      rpl_vol->compute_dew_volume (ct_vol->gpuit_float(), dew_vol, background);
      plm_image_save_vol (out_fn, dew_vol);
    }

    if (parms->mode==2)  {
      Volume* aperture_vol;
      Volume* segdepth_vol;
      
      aperture_vol = create_segdepth_volume (parms, scene);
      segdepth_vol = create_segdepth_volume (parms, scene);
      rpl_vol->compute_segdepth_volume (ct_vol->gpuit_float(), aperture_vol, segdepth_vol, background);
      plm_image_save_vol (out_fn, segdepth_vol);
      plm_image_save_vol (parms->output_ap_fn.c_str(), aperture_vol);
    }
}

int
wed_ct_initialize(Wed_Parms *parms)
{
  

  
  Plm_image* ct_vol;
  Plm_image* dose_vol = 0;
  Proton_Scene scene;
  
  
  float background[3];
  //Background value for wed ct output
  background[0] = -1000.;
  //Background value for wed dose output
  background[1] = 0.;
 //Background value for radiation length output
  background[2] = 0.;

  /* load the patient and insert into the scene */
  ct_vol = plm_image_load (parms->input_ct_fn, PLM_IMG_TYPE_ITK_FLOAT);
  if (!ct_vol) {
    fprintf (stderr, "\n** ERROR: Unable to load patient volume.\n");
    return -1;
  }
  
  if (parms->skin_fn != "") {
    fprintf (stderr, "\n** Skin file defined.  Modifying input ct...\n");
 
    Volume* ct_volume = ct_vol->gpuit_float();
    Plm_image* skin_vol = plm_image_load (parms->skin_fn, PLM_IMG_TYPE_ITK_FLOAT);
    if (!skin_vol) {
      fprintf (stderr, "\n** ERROR: Unable to load skin input.\n");
      return -1;
    }
    Volume* skin_volume = skin_vol->gpuit_float();
    
    if (skin_ct(ct_volume, skin_volume, background[0]))  {  //apply skin input to ct
      fprintf (stderr, "\n** ERROR: Unable to apply skin input to ct input.\n");
    }

  }


  
  scene.set_patient (ct_vol);
  
  printf("%s\n",parms->input_dose_fn.c_str());
 
  //  if (parms->input_dose_fn != "" && parms->output_dose_fn != "") {
    //Load the input dose, or input wed_dose
    dose_vol = plm_image_load (parms->input_dose_fn.c_str(), 
			       PLM_IMG_TYPE_ITK_FLOAT);
  
  /* set scene parameters */
  scene.beam->set_source_position (parms->src);
  scene.beam->set_isocenter_position (parms->isocenter);
  
  scene.ap->set_distance (parms->ap_offset);
  scene.ap->set_spacing (parms->ap_spacing);
  
  //Scene dimensions are set by .cfg file

  //Note: Set dimensions first, THEN center, as set_dim() also changes center
  int ap_res[2];
  float ap_center[2];

  //Aperture dimensions
  if (parms->have_ires) {scene.ap->set_dim (parms->ires);}
  //If dew option, and not specified in .cfg files, then we guess
  //at some scene dimensions set by input wed image.
  if (parms->mode==1)  {
   if (!parms->have_ires)  {
      Volume *wed_vol = dose_vol->gpuit_float();
      //Grab aperture dimensions from input wed.
      //We also pad each dimension by 1, for the later trilinear interpolations.
      ap_res[0] = (int) (wed_vol->dim[0]+2);
      ap_res[1] = (int) (wed_vol->dim[1]+2);
      scene.ap->set_dim (ap_res);
      parms->ires[0]=ap_res[0];
      parms->ires[1]=ap_res[1];
    }
  }

  //Aperture Center
  //Note - Center MUST be reset here if set in the config file, as set_dim()
  //will reset the center.
  if (parms->have_ic) {scene.ap->set_center (parms->ic);}
  //And again, guess.
  if (parms->mode==1)  {
    if (!parms->have_ic)  {
      //Set center as half the resolutions.
      ap_center[0] = (float) ap_res[0]/2.;
      ap_center[1] = (float) ap_res[1]/2.;
      scene.ap->set_center (ap_center);
       parms->ic[0]=ap_center[0];
      parms->ic[1]=ap_center[1];
    } 
  }

  scene.set_step_length(parms->ray_step);
  
  /* try to setup the scene with the provided parameters */
  if (!scene.init ()) {
    fprintf (stderr, "ERROR: Unable to initilize scene.\n");
    return -1;
  }
  scene.debug ();
  
  if (parms->rpl_vol_fn != "") {
    scene.rpl_vol->save (parms->rpl_vol_fn);
  }

  printf ("Working...\n");
  fflush(stdout);
  
  if (parms->mode==0)  {
    printf ("Computing patient wed volume...\n");
    wed_ct_compute (parms->output_ct_fn, parms, ct_vol, &scene, background[0]);
    printf ("done.\n");
  }
  
  if (parms->input_dose_fn != "" && parms->output_dose_fn != "") {
    if ((parms->mode==0)||(parms->mode==1))  {
      printf ("Calculating dose...\n");
      wed_ct_compute (parms->output_dose_fn.c_str(), 
		      parms, dose_vol, &scene, background[1]);
      printf ("Complete...\n");
    }
  }
  if (parms->mode==2)  {
    printf ("Calculating depths...\n");
    wed_ct_compute (parms->output_depth_fn.c_str(), 
		    parms, dose_vol, &scene, background[2]);
    printf ("Complete...\n");
  }


  return 0;
}

int
main (int argc, char* argv[])
{

  Wed_Parms *parms = new Wed_Parms();
  int wed_iter = 1;
  
  if (!parms->parse_args (argc, argv)) { //sets parms if input with .cfg file, skips with group option
    exit (0); 
  }
  
  if (parms->group)  {
    wed_iter = 0;
    
    while(wed_iter!=parms->group)  {
      if (parms->group) {
	parms->parse_group(argc, argv, wed_iter);
	wed_ct_initialize(parms);
	wed_iter++;
      }
      
    }
  }
    
  else {wed_ct_initialize(parms);} //Compute wed without loop
  
  return 0;
}
