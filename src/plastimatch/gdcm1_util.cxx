/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmBinEntry.h"
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"

#include "gdcm_dose.h"
#include "gdcm_series.h"
#include "logfile.h"
#include "itk_image_stats.h"
#include "img_metadata.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_image_type.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "referenced_dicom_dir.h"
#include "rtss_polyline_set.h"
#include "volume.h"


void
set_metadata_from_gdcm_file (
    Img_metadata *img_metadata, 
    /* const */ gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
)
{
    std::string tmp = gdcm_file->GetEntryValue (group, elem);
    if (tmp != gdcm::GDCM_UNFOUND) {
	img_metadata->set_metadata (group, elem, tmp);
    }
}

void
set_gdcm_file_from_metadata (
    gdcm::File *gdcm_file, 
    const Img_metadata *img_metadata, 
    unsigned short group, 
    unsigned short elem
)
{
    gdcm_file->InsertValEntry (
	img_metadata->get_metadata (group, elem), group, elem);
}
