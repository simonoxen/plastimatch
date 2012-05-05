/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "itkImage.h"
#include "getopt.h"

#include "plmbase.h"

#include "plm_image_header.h"
#include "vf_invert.h"

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

#define MAX_ITS 20

#if defined (commentout)
void
vf_invert_main (Vf_Invert_Parms* parms)
{
    typedef itk::InverseDeformationFieldImageFilter < DeformationFieldType, DeformationFieldType >  FilterType;
    
    Plm_image_header pih;

    if (parms->fixed_img_fn[0]) {
        /* if given, use the parameters from user-supplied fixed image */
        FloatImageType::Pointer fixed = load_float (parms->fixed_img_fn);
        pih.set_from_itk_image (fixed);
    } else {
        pih.set_from_gpuit (parms->origin, parms->spacing, parms->dim);
    }

    FilterType::Pointer filter = FilterType::New ();
    DeformationFieldType::Pointer vf_in = load_float_field (parms->vf_in_fn);
    filter->SetInput (vf_in);
    filter->SetOutputOrigin (pih.m_origin);
    filter->SetOutputSpacing (pih.m_spacing);
    filter->SetSize (pih.m_region.GetSize());

    //filter->SetOutsideValue( 0 );
    filter->Update();
    DeformationFieldType::Pointer vf_out = filter->GetOutput();
    save_image (vf_out, parms->vf_out_fn);
}
#endif

void
vf_invert_main (Vf_Invert_Parms* parms)
{
    plm_long i, j, k, v;
    int its;
    float x, y, z;
    Plm_image_header pih;
    Volume *mask, *vf_in, *vf_inv, *vf_smooth, *vf_out;
    float *img_in, *img_inv, *img_smooth, *img_out;
    unsigned char *img_mask;
    float ker[3] = { 0.3, 0.4, 0.3 };

    if (parms->fixed_img_fn[0]) {
        /* if given, use the parameters from user-supplied fixed image */
        FloatImageType::Pointer fixed = itk_image_load_float (
            parms->fixed_img_fn, 0);
        pih.set_from_itk_image (fixed);
        pih.get_origin (parms->origin);
        pih.get_spacing (parms->spacing);
        pih.get_dim (parms->dim);

        pih.set_from_gpuit (parms->dim, parms->origin, parms->spacing, 0);
    }

    /* GCS FIX: Need direction cosines */
    /* Create mask volume */
    mask = new Volume (parms->dim, parms->origin, parms->spacing, 0,
        PT_UCHAR, 1);

    /* GCS FIX: Need direction cosines */
    /* Create tmp volume */
    vf_inv = new Volume (parms->dim, parms->origin, parms->spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 1);

    /* Load input vf */
    vf_in = read_mha (parms->vf_in_fn);
    vf_convert_to_interleaved (vf_in);

    /* Populate mask & tmp volume */
    img_mask = (unsigned char*) mask->img;
    img_in = (float*) vf_in->img;
    img_inv = (float*) vf_inv->img;
    for (z = vf_in->offset[2], k = 0, v = 0; k < vf_in->dim[2]; k++, z+=vf_in->spacing[2]) {
        for (y = vf_in->offset[1], j = 0; j < vf_in->dim[1]; j++, y+=vf_in->spacing[1]) {
            for (x = vf_in->offset[0], i = 0; i < vf_in->dim[0]; v++, i++, x+=vf_in->spacing[0]) {
                plm_long mijk[3], midx;
                float mxyz[3];
                mxyz[0] = x + img_in[3*v+0];
                mijk[0] = round_int ((mxyz[0] - vf_inv->offset[0]) / vf_inv->spacing[0]);
                mxyz[1] = y + img_in[3*v+1];
                mijk[1] = (mxyz[1] - vf_inv->offset[1]) / vf_inv->spacing[1];
                mxyz[2] = z + img_in[3*v+2];
                mijk[2] = (mxyz[2] - vf_inv->offset[2]) / vf_inv->spacing[2];

                if (mijk[0] < 0 || mijk[0] >= vf_inv->dim[0]) continue;
                if (mijk[1] < 0 || mijk[1] >= vf_inv->dim[1]) continue;
                if (mijk[2] < 0 || mijk[2] >= vf_inv->dim[2]) continue;

                midx = (mijk[2] * vf_inv->dim[1] + mijk[1]) * vf_inv->dim[0] + mijk[0];
                img_inv[3*midx+0] = -img_in[3*v+0];
                img_inv[3*midx+1] = -img_in[3*v+1];
                img_inv[3*midx+2] = -img_in[3*v+2];
                img_mask[midx] ++;
            }
        }
    }

    /* We're done with input volume now. */
    delete vf_in;

    /* GCS FIX: Need direction cosines */
    /* Create tmp & output volumes */
    vf_out = new Volume (parms->dim, parms->origin, parms->spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 3);
    img_out = (float*) vf_out->img;
    /* GCS FIX: Need direction cosines */
    vf_smooth = new Volume (parms->dim, parms->origin, parms->spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 3);
    img_smooth = (float*) vf_smooth->img;

    /* Iterate, pasting and smoothing */
    printf ("Paste and smooth loop\n");
    for (its = 0; its < MAX_ITS; its++) {
        printf ("Iteration %d/%d\n", its, MAX_ITS);
        /* Paste */
        for (v = 0, k = 0; k < vf_out->dim[2]; k++) {
            for (j = 0; j < vf_out->dim[1]; j++) {
                for (i = 0; i < vf_out->dim[0]; i++, v++) {
                    if (img_mask[v]) {
                        img_smooth[3*v+0] = img_inv[3*v+0];
                        img_smooth[3*v+1] = img_inv[3*v+1];
                        img_smooth[3*v+2] = img_inv[3*v+2];
                    } else {
                        img_smooth[3*v+0] = img_out[3*v+0];
                        img_smooth[3*v+1] = img_out[3*v+1];
                        img_smooth[3*v+2] = img_out[3*v+2];
                    }
                }
            }
        }

        /* Smooth the estimate into vf_out.  The volumes are ping-ponged. */
        printf ("Convolving\n");
        vf_convolve_x (vf_out, vf_smooth, ker, 3);
        vf_convolve_y (vf_smooth, vf_out, ker, 3);
        vf_convolve_z (vf_out, vf_smooth, ker, 3);
    }
    printf ("Done.\n");

    /* We're done with the mask & smooth image. */
    delete mask;
    delete vf_smooth;

    /* Write the output */
    write_mha (parms->vf_out_fn, vf_out);
    delete vf_out;
}

void
print_usage (void)
{
    printf ("Usage: vf_invert --input=vf_in --output=vf_out\n");
    printf ("           --dims=\"x y z\" --origin=\"x y z\" --spacing=\"x y z\"\n");
    printf ("       ||  --fixed=\"fixed-img\"\n");
    exit (-1);
}

void
parse_args (Vf_Invert_Parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
        { "input",          required_argument,      NULL,           1 },
        { "output",         required_argument,      NULL,           2 },
        { "dims",           required_argument,      NULL,           3 },
        { "origin",         required_argument,      NULL,           4 },
        { "spacing",        required_argument,      NULL,           5 },
        { "fixed",          required_argument,      NULL,           6 },
        { NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
        switch (ch) {
        case 1:
            strncpy (parms->vf_in_fn, optarg, _MAX_PATH);
            break;
        case 2:
            strncpy (parms->vf_out_fn, optarg, _MAX_PATH);
            break;
        case 3: {
            int d[3];
            rc = sscanf (optarg, "%d %d %d", &d[0], &d[1], &d[2]);
            if (rc != 3) {
                print_usage();
            }
            parms->dim[0] = d[0];
            parms->dim[1] = d[1];
            parms->dim[2] = d[2];
        }
            break;
        case 4:
            rc = sscanf (optarg, "%g %g %g", &(parms->origin[0]), 
                &(parms->origin[1]), &(parms->origin[2]));
            if (rc != 3) {
                print_usage();
            }
            break;
        case 5:
            rc = sscanf (optarg, "%g %g %g", &(parms->spacing[0]), 
                &(parms->spacing[1]), &(parms->spacing[2]));
            if (rc != 3) {
                print_usage();
            }
            break;
        case 6:
            strncpy (parms->fixed_img_fn, optarg, _MAX_PATH);
            break;
        default:
            break;
        }
    }
    if (!parms->vf_in_fn[0] || !parms->vf_out_fn[0] || ((!parms->dim[0] || !parms->origin[0] || !parms->spacing[0]) && (!parms->fixed_img_fn[0]))) {
        printf ("Error: must specify all options\n");
        print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Vf_Invert_Parms parms;
    
    parse_args (&parms, argc, argv);

    vf_invert_main (&parms);

    printf ("Finished!\n");
    return 0;
}
