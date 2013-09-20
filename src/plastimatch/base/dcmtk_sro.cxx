/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_module_general_study.h"
#include "dcmtk_module_patient.h"
#include "dcmtk_sro.h"
#include "file_util.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "xform.h"

void
Dcmtk_sro::save (
    Xform* xf,
    const Rt_study_metadata::Pointer& rtm_src,
    const Rt_study_metadata::Pointer& rtm_reg,
    const std::string& dicom_dir)
{
    Xform xf_aff;
    xform_to_aff (&xf_aff, xf, 0);

    AffineTransformType::Pointer itk_aff = xf_aff.get_aff();

    /* Prepare output file */
    std::string sro_fn = string_format ("%s/sro.dcm", dicom_dir.c_str());
    make_directory_recursive (sro_fn);

    /* Prepare dcmtk */
    OFCondition ofc;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    Metadata *study_meta = 0;
    if (rtm_src) {
        study_meta = rtm_src->get_study_metadata ();
    } else if (rtm_reg) {
        study_meta = rtm_reg->get_study_metadata ();
    }


    Dcmtk_module_patient::set (dataset, study_meta);
    Dcmtk_module_general_study::set (dataset, study_meta);

    /* ----------------------------------------------------------------- */
    /*     Write the output file                                         */
    /* ----------------------------------------------------------------- */
    ofc = fileformat.saveFile (sro_fn.c_str(), EXS_LittleEndianExplicit);
    if (ofc.bad()) {
        print_and_exit (
            "Error: cannot write DICOM Spatial Registration (%s)\n", 
            ofc.text());
    }
}
