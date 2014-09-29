#include "volume.h"
#include "wed_parms.h"

//Should eventually be made into a class - for now, just some clean up.

Volume* create_wed_volume (Wed_Parms* parms, Rpl_volume* rpl_vol)
{

   /* water equivalent depth volume has the same x,y dimensions as the rpl
     * volume. Note: this means the wed x,y dimensions are equal to the
     * aperture dimensions and the z-dimension is equal to the sampling
     * resolution chosen for the rpl */
    plm_long wed_dims[3];

    Volume *vol = rpl_vol->get_vol ();
    wed_dims[0] = vol->dim[0];
    wed_dims[1] = vol->dim[1];
    wed_dims[2] = vol->dim[2];

    /////////////////////////////
    //Should be insenstive to aperture rotation?
    /*
    Proj_volume *proj_vol = d_ptr->proj_vol;
    double iso_src_vec[3];   //vector from isocenter to source
    proj_vol->get_proj_matrix()->get_nrm(iso_src_vec);
    */

    float xoff = -(vol->dim[0] - parms->ic[0]);
    float yoff = -(vol->dim[1] - parms->ic[1]);

    float wed_off[3] = {xoff, yoff, 0.0f};
    float wed_ps[3] = {1.0f, 1.0f, 1.0f};

    return new Volume (wed_dims, wed_off, wed_ps, NULL, PT_FLOAT, 1);
}

static Volume*
create_dew_volume (Wed_Parms* parms, Ion_plan *scene)
{
    Volume::Pointer patient_vol = scene->get_patient_volume();

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

Volume* create_proj_wed_volume (Rpl_volume* rpl_vol)
{
    float proj_wed_off[3] = {0.0f, 0.0f, 0.0f};
    float proj_wed_ps[3] = {1.0f, 1.0f, 1.0f};
    plm_long proj_wed_dims[3];

    Volume *vol = rpl_vol->get_vol ();
    proj_wed_dims[0] = vol->dim[0];
    proj_wed_dims[1] = vol->dim[1];
    proj_wed_dims[2] = 1;

    return new Volume (proj_wed_dims, proj_wed_off, proj_wed_ps, NULL, PT_FLOAT, 1);
}

Volume* create_proj_sinogram_volume (Wed_Parms* parms, Volume *proj_wed_vol)
{
    float proj_wed_off[3];
    proj_wed_off[0] = proj_wed_vol->offset[0];
    proj_wed_off[1] = proj_wed_vol->offset[1];
    proj_wed_off[2] = proj_wed_vol->offset[2];

    float proj_wed_ps[3];
    proj_wed_ps[0] = proj_wed_vol->spacing[0];
    proj_wed_ps[1] = proj_wed_vol->spacing[1];
    proj_wed_ps[2] = proj_wed_vol->spacing[2];

    plm_long proj_wed_dims[3];
    proj_wed_dims[0] = proj_wed_vol->dim[0];
    proj_wed_dims[1] = proj_wed_vol->dim[1];
    proj_wed_dims[2] = parms->sinogram_res;

    return new Volume (proj_wed_dims, proj_wed_off, proj_wed_ps, NULL, PT_FLOAT, 1);
}


