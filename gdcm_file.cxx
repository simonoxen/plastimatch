/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"

/* This is a wrapper for gdcm::File, in order to avoid problems with 
   their buggy implementation of stdint.h typedefs */

std::string
gdcm_file_GetEntryValue (gdcm::File *file, unsigned short group, 
    unsigned short elem)
{
    return file->GetEntryValue (group, elem);
}

const std::string&
gdcm_file_GDCM_UNKNOWN ()
{
    return gdcm::GDCM_UNKNOWN;
}

const std::string&
gdcm_file_GDCM_UNFOUND ()
{
    return gdcm::GDCM_UNFOUND;
}
