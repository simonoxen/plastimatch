/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#include "gdcm1_rtss.h"
#include "gdcm1_series.h"
#endif
#include "logfile.h"
#include "print_and_exit.h"
#include "rtds_dicom.h"
#include "rtss.h"

void
Rtds::load_dicom (const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_dir);
#else
    this->load_gdcm (dicom_dir);
#endif
}
