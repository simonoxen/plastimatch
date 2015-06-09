/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "print_and_exit.h"
#include "string_util.h"
#include "xio_demographic.h"

Xio_demographic::Xio_demographic (const char *filename)
{
    std::ifstream ifs (filename);
    if (ifs.fail()) {
        print_and_exit ("Error opening file %s for read\n", filename);
    }

    /* version string 
       00011017 - pxio version 4.2 */
    std::string version;
    getline (ifs, version);

    /* date (for what?) */
    std::string date;
    getline (ifs, date);

    /* important stuff here */
    getline (ifs, m_patient_name);
    m_patient_name = string_trim (m_patient_name);
    getline (ifs, m_patient_id);
    m_patient_id = string_trim (m_patient_id);
}

Xio_demographic::~Xio_demographic ()
{
}
