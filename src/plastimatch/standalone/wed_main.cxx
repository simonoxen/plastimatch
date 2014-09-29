/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "aperture.h"
#include "ion_beam.h"
#include "ion_plan.h"
#include "plm_image.h"
#include "plm_math.h"
#include "proj_volume.h"
#include "ray_trace_probe.h"
#include "rpl_volume.h"
#include "volume.h"
#include "volume_limit.h"
#include "wed_parms.h"

#include "create_wed_volumes.cxx"


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

void
wed_ct_compute (
    const char* out_fn,
    Wed_Parms* parms,
    Plm_image::Pointer& ct_vol,  // This is not always ct, 
                                 //  sometimes it is dose or 
                                 //  sometimes it is target mask.
    Ion_plan *scene,
    float background
)
{
    Rpl_volume* rpl_vol = scene->rpl_vol;

    if (parms->mode==0)  {
        Volume* wed_vol;
	wed_vol = create_wed_volume (parms, rpl_vol);
        rpl_vol->compute_wed_volume (wed_vol, ct_vol->get_volume_float().get(), 
            background);
        Plm_image(wed_vol).save_image(out_fn);
    }

    if (parms->mode==1)  {
        Volume* dew_vol;
	//Fix below function, move to rpl_volume as create_wed_volume above.
	//Dew parameters will need to be incorporated into ion_scene
        dew_vol = create_dew_volume (parms, scene);
        rpl_vol->compute_dew_volume (ct_vol->get_volume_float().get(), 
            dew_vol, background);
        Plm_image(dew_vol).save_image(out_fn);
    }

    if (parms->mode==2) {
        /* Compute the aperture and range compensator */
        rpl_vol->compute_beam_modifiers (
            ct_vol->get_volume_float().get(), 
            background);

        /* Save files as output */
        Plm_image::Pointer& ap 
            = rpl_vol->get_aperture()->get_aperture_image();
        Plm_image::Pointer& rc 
            = rpl_vol->get_aperture()->get_range_compensator_image();

        ap->save_image (parms->output_ap_fn.c_str());
        rc->save_image (out_fn);
    }

    if (parms->mode==3)  {
        Volume* proj_wed_vol;
        Volume* sinogram_vol;
	
        proj_wed_vol = create_proj_wed_volume(rpl_vol);

	if (parms->sinogram!=0)  {
            sinogram_vol = create_proj_sinogram_volume(parms, proj_wed_vol);

            float *sin_img = (float*) sinogram_vol->img;
            float *proj_img = (float*) proj_wed_vol->img;
            plm_long ijk[3];
            plm_long n_voxels_sin = sinogram_vol->dim[0]*sinogram_vol->dim[1]*sinogram_vol->dim[2];
            plm_long n_voxels_proj = proj_wed_vol->dim[0]*proj_wed_vol->dim[1]*proj_wed_vol->dim[2];
            float *sin_array = new float[n_voxels_sin];
            float *proj_array = new float[n_voxels_proj];

            //Loop over angles determined by the resolution, and calcaulate a projective
            //volume for each.  Then fill the sinogram volume with each slice.
            int angles = parms->sinogram_res;

            float src[3] = {parms->src[0], parms->src[1], parms->src[2]};
            float iso[3] = {parms->isocenter[0], parms->isocenter[1], parms->isocenter[2]};

            float src2[3] = {0, 0, parms->src[2]};
            float radius[2] = {src[0]-iso[0], src[1]-iso[1]};
            float radius_len = sqrt( (src[0]-iso[0])*(src[0]-iso[0]) + (src[1]-iso[1])*(src[1]-iso[1]));

            float init_angle = atan2(radius[1],radius[0]);
            float angle = 0;

            for (int i=0; i!=angles; ++i)  {
                angle = init_angle + ( i / (float) parms->sinogram_res)*2.*M_PI;
                src2[0] = cos(angle)*radius_len + iso[0];
                src2[1] = sin(angle)*radius_len + iso[1];

                scene->beam->set_source_position (src2);
		scene->init();
                rpl_vol = scene->rpl_vol;
                rpl_vol->compute_proj_wed_volume (proj_wed_vol, background);

                //Fill proj array with voxel values.
                for (plm_long zz=0; zz!=n_voxels_proj; ++zz)  {
                    proj_array[zz] = proj_img[zz];
                }
	
                for (int j=0; j!=proj_wed_vol->dim[0]; ++j)  {
                    for (int k=0; k!=proj_wed_vol->dim[1]; ++k)  {
                        sin_array[sinogram_vol->index(j,k,i)]
                            = proj_array[proj_wed_vol->index(j,k,0)];
                    }
                }
            }
            //Fill sinogram image with voxel values from assembled array.
            for (plm_long zz=0; zz!=n_voxels_sin; ++zz)  {
                sin_img[zz] = sin_array[zz];
            }

            Plm_image(sinogram_vol).save_image(out_fn);

            delete[] sin_array;
            delete[] proj_array;
	}

	else {
            rpl_vol->compute_proj_wed_volume (proj_wed_vol, background);
            Plm_image(proj_wed_vol).save_image(out_fn);
	}       
    }
}

void
wed_ct_compute (
    const char* out_fn,
    Wed_Parms* parms,
    Ion_plan *scene,
    float background
)
{
    Plm_image::Pointer ct_vol = Plm_image::New();
    wed_ct_compute (out_fn, parms, ct_vol, scene, background);
}

int
wed_ct_initialize(Wed_Parms *parms)
{

    Plm_image::Pointer dose_vol;
    Ion_plan scene;
    float background[4];

    /*
    for (int i=0; i=100000000;++i)  {
      scene.init ();
    }
    */

    //Background value for wed ct output
    background[0] = -1000.;
    //Background value for wed dose output
    background[1] = 0.;
    //Background value for radiation length output
    background[2] = 0.;
    //Background value for projection of wed
    background[3] = 0.;

    /* load the patient and insert into the scene */
    Plm_image::Pointer ct_vol = Plm_image::New (
        parms->input_ct_fn, PLM_IMG_TYPE_ITK_FLOAT);
    if (!ct_vol) {
        fprintf (stderr, "\n** ERROR: Unable to load patient volume.\n");
        return -1;
    }
  
    if (parms->skin_fn != "") {
        fprintf (stderr, "\n** Skin file defined.  Modifying input ct...\n");
 
        Volume* ct_volume = ct_vol->get_volume_float().get();
        Plm_image::Pointer skin_vol = plm_image_load (parms->skin_fn, 
            PLM_IMG_TYPE_ITK_FLOAT);
        if (!skin_vol) {
            fprintf (stderr, "\n** ERROR: Unable to load skin input.\n");
            return -1;
        }
        Volume* skin_volume = skin_vol->get_volume_float().get();
    
        if (skin_ct(ct_volume, skin_volume, background[0]))  {
            //apply skin input to ct
            fprintf (stderr, "\n** ERROR: Unable to apply skin input to ct input.\n");
        }
    }
  
    scene.set_patient (ct_vol);
  
    printf("%s\n",parms->input_dose_fn.c_str());
 
    //  if (parms->input_dose_fn != "" && parms->output_dose_fn != "") {
    //Load the input dose, or input wed_dose
    if ((parms->mode==0)||(parms->mode==1))  {
        dose_vol = plm_image_load (parms->input_dose_fn.c_str(), 
            PLM_IMG_TYPE_ITK_FLOAT);
    }
    /* set scene parameters */
    scene.beam->set_source_position (parms->src);
    scene.beam->set_isocenter_position (parms->isocenter);

    scene.get_aperture()->set_distance (parms->ap_offset);
    scene.get_aperture()->set_spacing (parms->ap_spacing);
  
    //Scene dimensions are set by .cfg file

    //Note: Set dimensions first, THEN center, as set_dim() also changes center
    int ap_res[2];
    float ap_center[2];

    //Aperture dimensions
    if (parms->have_ires) {
        scene.get_aperture()->set_dim (parms->ires);
	//If dew, pad each by one for interpolations
	if (parms->mode==1)  {
            ap_res[0] = (int) (parms->ires[0]+2);
            ap_res[1] = (int) (parms->ires[1]+2);
            scene.get_aperture()->set_dim (ap_res);
            parms->ires[0]=ap_res[0];
            parms->ires[1]=ap_res[1];
	}
    }
    //If dew option, and not specified in .cfg files, then we guess
    //at some scene dimensions set by input wed image.

    else if (parms->mode==1)  {
        Volume *wed_vol = dose_vol->get_volume_float().get();
        //Grab aperture dimensions from input wed.
        //We also pad each dimension by 1, for the later trilinear 
        //interpolations.
        ap_res[0] = (int) (wed_vol->dim[0]+2);
        ap_res[1] = (int) (wed_vol->dim[1]+2);
  
        scene.get_aperture()->set_dim (ap_res);
        parms->ires[0]=ap_res[0];
        parms->ires[1]=ap_res[1];
    }

    //Aperture Center
    //Note - Center MUST be reset here if set in the config file, as set_dim()
    //will reset the center.
    if (parms->have_ic) {

        if (parms->mode==1)  {

	  //If center is not defined in config file (in general,
	  //it shouldn't be), then default values should be reset
	  if ((parms->ic[0]==-99.5)&&(parms->ic[2]==-99.5))  {
	    Volume *wed_vol = dose_vol->get_volume_float().get();
	    parms->ic[0] = wed_vol->offset[0] + wed_vol->dim[0];
	    parms->ic[1] = wed_vol->offset[1] + wed_vol->dim[1];
	  }

	  //else set it at the center
	  else {
            ap_center[0] = parms->ic[0]+1.*parms->ap_spacing[0];
            ap_center[1] = parms->ic[1]+1.*parms->ap_spacing[1];
            scene.get_aperture()->set_center (ap_center);
	  }
        }

        else {
            ap_center[0] = parms->ic[0];
            ap_center[1] = parms->ic[1];
            scene.get_aperture()->set_center (ap_center);
        }

    }
    //And again, guess if not specified.
    //Note - add the dew option below if not specified.
    else if (parms->mode==1)  {
        //Set center as half the resolutions.
        ap_center[0] = (float) ap_res[0]/2.;
        ap_center[1] = (float) ap_res[1]/2.;
        scene.get_aperture()->set_center (ap_center);
        parms->ic[0]=ap_center[0];
        parms->ic[1]=ap_center[1];
    } 

    scene.set_step_length(parms->ray_step);



    /* Try to setup the scene with the provided parameters.
       This function computes the rpl volume. */
    if (!scene.init ()) {
        fprintf (stderr, "ERROR: Unable to initilize scene.\n");
        return -1;
    }

    scene.debug ();

    /* Save rpl volume if requested */
    if (parms->rpl_vol_fn != "") {
        scene.rpl_vol->save (parms->rpl_vol_fn);
    }

    printf ("Working...\n");
    fflush(stdout);
  
    /* Compute the ct-wed volume */
    if (parms->mode==0)  {
        printf ("Computing patient wed volume...\n");
        wed_ct_compute (parms->output_ct_fn, parms, ct_vol, &scene, background[0]);
        printf ("done.\n");
    }
  
    /* Compute the dose-wed volume */
    if (parms->input_dose_fn != "" && parms->output_dose_fn != "") {
        if ((parms->mode==0)||(parms->mode==1))  {
            printf ("Calculating dose...\n");
            wed_ct_compute (parms->output_dose_fn.c_str(), 
                parms, dose_vol, &scene, background[1]);
            printf ("Complete...\n");
        }
    }

    /* Compute the aperture and range compensator volumes */
    if (parms->mode==2)  {
        printf ("Calculating depths...\n");
        wed_ct_compute (parms->output_depth_fn.c_str(), 
            parms, dose_vol, &scene, background[2]);
        printf ("Complete...\n");
    }

    /* Compute the projective wed volume */
    if (parms->mode==3)  {
        printf ("Calculating wed projection...\n");
        wed_ct_compute (parms->output_proj_wed_fn.c_str(), 
            parms, &scene, background[3]);
        printf ("Complete...\n");
    }

    return 0;
}

int
main (int argc, char* argv[])
{
    Wed_Parms *parms = new Wed_Parms();
    int wed_iter = 1;
  
    //sets parms if input with .cfg file, skips with group option
    if (!parms->parse_args (argc, argv)) {
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
    else {
        //Compute wed without loop
        wed_ct_initialize(parms);
    }
  
    return 0;
}
