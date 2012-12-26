/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "geometry_chooser.h"
#include "itk_image_load.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "vf_convolve.h"
#include "vf_invert.h"
#include "volume.h"
#include "volume_header.h"
#include "xform.h"

class Vf_invert_private {
public:
    Vf_invert_private () {
        iterations = 20;
        vf_out = 0;
    }
    ~Vf_invert_private () {
        delete vf_out;
    }
public:
    int iterations;
    Geometry_chooser gchooser;
    DeformationFieldType::Pointer input_vf;
    Volume *vf_out;
};

Vf_invert::Vf_invert () {
    this->d_ptr = new Vf_invert_private;
}

Vf_invert::~Vf_invert () {
    delete this->d_ptr;
}

void 
Vf_invert::set_input_vf (const char* vf_fn)
{
    d_ptr->input_vf = itk_image_load_float_field (vf_fn);
    d_ptr->gchooser.set_reference_image (d_ptr->input_vf);
}

void 
Vf_invert::set_input_vf (
    const DeformationFieldType::Pointer vf)
{
    d_ptr->input_vf = vf;
    d_ptr->gchooser.set_reference_image (d_ptr->input_vf);
}

void 
Vf_invert::set_fixed_image (const char* image_fn)
{
    d_ptr->gchooser.set_fixed_image (image_fn);
}

void 
Vf_invert::set_fixed_image (
    const FloatImageType::Pointer image)
{
    d_ptr->gchooser.set_fixed_image (image);
}

void 
Vf_invert::set_dim (const plm_long dim[3])
{
    d_ptr->gchooser.set_dim (dim);
}

void 
Vf_invert::set_origin (const float origin[3])
{
    d_ptr->gchooser.set_origin (origin);
}

void 
Vf_invert::set_spacing (const float spacing[3])
{
    d_ptr->gchooser.set_spacing (spacing);
}

void 
Vf_invert::set_direction_cosines (const float direction_cosines[9])
{
    d_ptr->gchooser.set_direction_cosines (direction_cosines);
}

void 
Vf_invert::run ()
{
    /* Compute geometry of output volume */
    const Plm_image_header *pih = d_ptr->gchooser.get_geometry ();
    Volume_header vh = pih->get_volume_header();

    /* Create mask volume */
    Volume *mask = new Volume (vh, PT_UCHAR, 1);

    /* Create tmp volume */
    Volume *vf_inv = new Volume (vh, PT_VF_FLOAT_INTERLEAVED, 1);

    /* Convert input vf to native, interleaved format */
    Xform *xf = new Xform;
    xf->set_itk_vf (d_ptr->input_vf);
    Volume *vf_in = xf->get_gpuit_vf ();
    vf_convert_to_interleaved (vf_in);

    /* Populate mask & tmp volume */
    unsigned char *img_mask = (unsigned char*) mask->img;
    float *img_in = (float*) vf_in->img;
    float *img_inv = (float*) vf_inv->img;

#pragma omp parallel for 
    LOOP_Z_OMP (k, vf_in) {
//    for (z = vf_in->offset[2], k = 0, v = 0; k < vf_in->dim[2]; k++, z+=vf_in->spacing[2]) {
//        for (y = vf_in->offset[1], j = 0; j < vf_in->dim[1]; j++, y+=vf_in->spacing[1]) {
//            for (x = vf_in->offset[0], i = 0; i < vf_in->dim[0]; v++, i++, x+=vf_in->spacing[0]) {

        plm_long fijk[3];      /* Index within fixed image (vox) */
        float fxyz[3];         /* Position within fixed image (mm) */
        fijk[2] = k;
        fxyz[2] = vf_in->offset[2] + fijk[2] * vf_in->step[2][2];
        LOOP_Y (fijk, fxyz, vf_in) {
            LOOP_X (fijk, fxyz, vf_in) {
                plm_long mijk[3], midx;
                float mxyz[3];
                plm_long v = volume_index (vf_in->dim, fijk);
                mxyz[0] = fxyz[0] + img_in[3*v+0];
                mijk[0] = ROUND_INT ((mxyz[0] - vf_inv->offset[0]) / vf_inv->spacing[0]);
                mxyz[1] = fxyz[1] + img_in[3*v+1];
                mijk[1] = (mxyz[1] - vf_inv->offset[1]) / vf_inv->spacing[1];
                mxyz[2] = fxyz[2] + img_in[3*v+2];
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
    delete xf;

    /* Create tmp & output volumes */
    Volume *vf_out = new Volume (vh, PT_VF_FLOAT_INTERLEAVED, 3);
    float *img_out = (float*) vf_out->img;
    Volume *vf_smooth = new Volume (vh, PT_VF_FLOAT_INTERLEAVED, 3);
    float *img_smooth = (float*) vf_smooth->img;

    /* Iterate, pasting and smoothing */
    printf ("Paste and smooth loop\n");
    for (int it = 0; it < d_ptr->iterations; it++) {
        printf ("Iteration %d/%d\n", it, d_ptr->iterations);
        /* Paste */
        for (plm_long v = 0, k = 0; k < vf_out->dim[2]; k++) {
            for (plm_long j = 0; j < vf_out->dim[1]; j++) {
                for (plm_long i = 0; i < vf_out->dim[0]; i++, v++) {
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
        float ker[3] = { 0.3, 0.4, 0.3 };
        printf ("Convolving\n");
        vf_convolve_x (vf_out, vf_smooth, ker, 3);
        vf_convolve_y (vf_smooth, vf_out, ker, 3);
        vf_convolve_z (vf_out, vf_smooth, ker, 3);
    }
    printf ("Done.\n");

    /* We're done with all these images now */
    delete mask;
    delete vf_inv;
    delete vf_smooth;

    /* Save the output image! */
    d_ptr->vf_out = vf_out;
}

const Volume*
Vf_invert::get_output_volume ()
{
    return d_ptr->vf_out;
}

/* ---------------------------------------------------------------------- */
#if defined (commentout)

#define MAX_ITS 20

class Vf_invert_parms {
public:
    std::string vf_in_fn;
    std::string vf_out_fn;
    std::string fixed_img_fn;
    float origin[3];
    float spacing[3];
    plm_long dim[3];
public:
    Vf_invert_parms () {
        vf_in_fn = "";
        vf_out_fn = "";
        fixed_img_fn = "";
        for (int d = 0; d < 3; d++) {
            origin[d] = 0.f;
            spacing[d] = 1.f;
            dim[d] = 0;
        }
    }
};

#if defined (commentout)
void
vf_invert_itk (Vf_Invert_Parms* parms)
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
    filter->SetOutputOrigin (pih.origin);
    filter->SetOutputSpacing (pih.spacing);
    filter->SetSize (pih.m_region.GetSize());

    //filter->SetOutsideValue( 0 );
    filter->Update();
    DeformationFieldType::Pointer vf_out = filter->GetOutput();
    save_image (vf_out, parms->vf_out_fn);
}
#endif

void
do_vf_invert (Vf_invert_parms* parms)
{
    plm_long i, j, k, v;
    int its;
    float x, y, z;
    Plm_image_header pih;
    Volume *mask, *vf_in, *vf_inv, *vf_smooth, *vf_out;
    float *img_in, *img_inv, *img_smooth, *img_out;
    unsigned char *img_mask;
    float ker[3] = { 0.3, 0.4, 0.3 };

    if (parms->fixed_img_fn != "") {
        /* if given, use the parameters from user-supplied fixed image */
        FloatImageType::Pointer fixed = itk_image_load_float (
            parms->fixed_img_fn.c_str(), 0);
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
    vf_in = read_mha (parms->vf_in_fn.c_str());
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
                mijk[0] = ROUND_INT ((mxyz[0] - vf_inv->offset[0]) / vf_inv->spacing[0]);
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
    write_mha (parms->vf_out_fn.c_str(), vf_out);
    delete vf_out;
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: vf_invert [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Vf_invert_parms *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "input", 
	"input vector field file name", 1, "");
    parser->add_long_option ("", "output", 
	"output vector field file name", 1, "");
    parser->add_long_option ("", "dim", 
        "size of output vector field in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "origin", 
        "location of first voxel of output vector field in mm \"x y z\"", 
        1, "");
    parser->add_long_option ("", "spacing", 
        "voxel spacing of output vector field in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "fixed", 
        "fixed image (match output vector field size to this image)", 1, "");

    /* Parse the command line arguments */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an input file was given */
    if (!parser->option ("input")) {
        throw (dlib::error ("Error.  Please specify an input file "
                "using the --input options"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output")) {
        throw (dlib::error ("Error.  Please specify an output file "
                "using the --output options"));
    }

    /* Check that output dimensions are known */
    if (!parser->option ("dim") 
        || !parser->option("origin")
        || !parser->option("spacing"))
    {
        if (!parser->option ("fixed")) {
            throw (dlib::error ("Error.  Please specify either dim, origin, "
                    " and spacing -or- a fixed file"));
        }
    }

    /* Copy values into parameter struct */
    parms->vf_in_fn = parser->get_string("input");
    parms->vf_out_fn = parser->get_string("output");
    if (parser->option ("fixed")) {
        parms->fixed_img_fn = parser->get_string("fixed");
    }
    if (parser->option ("dim")) {
        parser->assign_plm_long_13 (parms->dim, "dim");
    }
    if (parser->option ("origin")) {
        parser->assign_float13 (parms->origin, "origin");
    }
    if (parser->option ("spacing")) {
        parser->assign_float13 (parms->spacing, "spacing");
    }
}

int
main (int argc, char *argv[])
{
    Vf_invert_parms parms;
    
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv);
    do_vf_invert (&parms);

    printf ("Finished!\n");
    return 0;
}
#endif
