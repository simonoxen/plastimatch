/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdlib.h>

#include "bspline_xform.h"
#include "dcmtk_sro.h"
#include "logfile.h"
#include "pcmd_xf_convert.h"
#include "plm_clp.h"
#include "print_and_exit.h"
#include "pstring.h"
#include "rt_study_metadata.h"
#include "volume_header.h"
#include "xform.h"
#include "xform_convert.h"

class Xf_convert_parms {
public:
    Pstring input_fn;
    std::string output_type;
    std::string output_fn;
    std::string output_dicom_dir;

    std::string source_rcs;
    std::string registered_rcs;

    /* Geometry options */
    bool m_have_dim;
    bool m_have_origin;
    bool m_have_spacing;
    Volume_header m_vh;

    bool m_have_grid_spacing;
    float m_grid_spacing[3];

    Xform_convert xfc;
public:
    Xf_convert_parms () {
        m_have_dim = false;
        m_have_origin = false;
        m_have_spacing = false;
        m_have_grid_spacing = false;
    }
};

void
set_output_xform_type (Xform_convert *xfc, const std::string& output_type)
{
    if (output_type == "vf") {
        xfc->m_xf_out_type = XFORM_ITK_VECTOR_FIELD;
    }
    else if (output_type == "bspline") {
        xfc->m_xf_out_type = XFORM_GPUIT_BSPLINE;
    }
    else if (output_type == "itk_bsp" || output_type == "itk_bspline") {
        xfc->m_xf_out_type = XFORM_ITK_BSPLINE;
    }
    else if (output_type == "none") {
        xfc->m_xf_out_type = XFORM_NONE;
    }
    else {
        print_and_exit ("Sorry, can't convert output type\n");
    }
}

static void
do_xf_convert (Xf_convert_parms *parms)
{
    Xform_convert *xfc = &parms->xfc;

    /* Set up inputs */
    xfc->m_xf_in = new Xform;
    xfc->m_xf_out = new Xform;
    xform_load (xfc->m_xf_in, parms->input_fn);
    set_output_xform_type (xfc, parms->output_type);

    /* Set grid spacing as needed */
    xfc->m_xf_in->get_grid_spacing (xfc->m_grid_spac);
    if (parms->m_have_grid_spacing) {
        for (int d = 0; d < 3; d++) {
            xfc->m_grid_spac[d] = parms->m_grid_spacing[d];
        }
    }
    if (xfc->m_xf_in->m_type == XFORM_GPUIT_BSPLINE) {
        Bspline_xform* bxf = xfc->m_xf_in->get_gpuit_bsp();
        printf ("vox_per_rgn = %u %u %u\n", 
            (unsigned int) bxf->vox_per_rgn[0],
            (unsigned int) bxf->vox_per_rgn[1],
            (unsigned int) bxf->vox_per_rgn[2]
        );
        printf ("grid_spac = %g %g %g\n", 
            bxf->grid_spac[0],
            bxf->grid_spac[1],
            bxf->grid_spac[2]
        );
        printf ("grid_spac = %g %g %g\n", 
            xfc->m_grid_spac[0],
            xfc->m_grid_spac[1],
            xfc->m_grid_spac[2]
        );
    }

    /* Set volume header as needed */
    xfc->m_xf_in->get_volume_header (&xfc->m_volume_header);
    if (parms->m_have_dim) {
        xfc->m_volume_header.set_dim (parms->m_vh.get_dim());
    }
    if (parms->m_have_origin) {
        xfc->m_volume_header.set_origin (parms->m_vh.get_origin());
    }
    if (parms->m_have_spacing) {
        xfc->m_volume_header.set_spacing (parms->m_vh.get_spacing());
    }
    
    /* Do conversion */
    /* GCS FIX: This is not quite right.  Probably one should be 
       allowed to run xf-convert e.g. on a B-spline, and regrid 
       the b-spline, with the expectation that the xform 
       type will be read from the input file.  */
    if (xfc->m_xf_out_type != XFORM_NONE) {
        printf ("about to xform_convert\n");
        xform_convert (xfc);
        printf ("did xform_convert\n");
    }

    /* Save output file */
    if (!parms->output_fn.empty()) {
        xfc->m_xf_out->save (parms->output_fn);
    }

    /* Save output dicom */
    if (!parms->output_dicom_dir.empty()) {
        /* Load referenced image sets */
#if PLM_DCM_USE_DCMTK
        Rt_study_metadata::Pointer rtm_src;
        Rt_study_metadata::Pointer rtm_reg;
        
        if (!parms->source_rcs.empty()) {
            lprintf ("Loading source...\n");
            rtm_src = Rt_study_metadata::load (parms->source_rcs);
        }
        if (!parms->registered_rcs.empty()) {
            lprintf ("Loading registered...\n");
            rtm_reg = Rt_study_metadata::load (parms->registered_rcs);
        }

        Dcmtk_sro::save (
            xfc->m_xf_out, rtm_src, rtm_reg, parms->output_dicom_dir);
#endif
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch xf-convert [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Xf_convert_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "input", 
        "Input xform filename (required)", 1, "");
    parser->add_long_option ("", "output", 
        "Output xform filename", 1, "");
    parser->add_long_option ("", "output-type", 
        "Type of xform to create (required), choose from "
        "{bspline, itk_bspline, vf}", 1, "none");
    parser->add_long_option ("", "output-dicom", 
        "Directory for output of dicom spatial registration IOD", 1, "");

    parser->add_long_option ("", "dim", 
        "Size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "origin", 
        "Location of first image voxel in mm \"x y z\"", 1, "");
    parser->add_long_option ("", "spacing", 
        "Voxel spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "grid-spacing", 
        "B-spline grid spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "nobulk", 
        "Omit bulk transform for itk_bspline", 0);

    /* DICOM spatial registration */
    parser->add_long_option ("", "source-rcs", 
        "Directory containing source reference coordinate system"
        " (i.e. moving image)", 
        1, "");
    parser->add_long_option ("", "registered-rcs", 
        "Directory containing registered reference coordinate system"
        " (i.e. fixed image)", 
        1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Input files */
    parser->check_required ("input");
    parms->input_fn = parser->get_string("input").c_str();

    /* Output files */
    if (parser->option ("output")) {
        parms->output_fn = parser->get_string("output");
    }
    if (parser->option ("output-dicom")) {
        parms->output_dicom_dir = parser->get_string("output-dicom");
    }
    if (parms->output_dicom_dir.empty() && parms->output_fn.empty()) {
        throw (dlib::error ("Error.  You must specify either the --output "
                "or the --output-dicom option"));
    }

    /* DICOM spatial registration */
    parms->source_rcs = parser->get_string("source-rcs");
    parms->registered_rcs = parser->get_string("registered-rcs");

    Xform_convert *xfc = &parms->xfc;

    /* Copy values into output struct */
    parms->output_type = parser->get_string("output-type").c_str();

    /* Geometry options */
    if (parser->option ("dim")) {
        parms->m_have_dim = true;
        parser->assign_plm_long_13 (parms->m_vh.get_dim(), "dim");
    }
    if (parser->option ("origin")) {
        parms->m_have_origin = true;
        parser->assign_float_13 (parms->m_vh.get_origin(), "origin");
    }
    if (parser->option ("spacing")) {
        parms->m_have_spacing = true;
        parser->assign_float_13 (parms->m_vh.get_spacing(), "spacing");
    }
    if (parser->option ("grid-spacing")) {
        parms->m_have_grid_spacing = true;
        parser->assign_float_13 (parms->m_grid_spacing, "grid-spacing");
    }

    if (parser->option ("nobulk")) {
        xfc->m_nobulk = true;
    }
}

void
do_command_xf_convert (int argc, char *argv[])
{
    Xf_convert_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_xf_convert (&parms);
}
