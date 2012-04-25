/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <algorithm>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "bstrlib.h"

#include "file_util.h"
#include "print_and_exit.h"
#include "xio_dir.h"
#include "xio_patient.h"
#include "xio_plan.h"

Xio_patient::Xio_patient (
    const char* path
) 
{
    this->m_path = path;
    this->m_demographic_fn = "";
    this->num_studyset_dir = 0;
    this->num_plan_dir = 0;
    this->studyset_dir = 0;
    this->plan_dir = 0;
}

Xio_patient::~Xio_patient () 
{
    /* GCS FIX: Let it leak... */
}

void
Xio_patient::add_studyset_dir (
    std::string studyset_path
)
{
    Xio_studyset_dir *xsd;

    this->studyset_dir = (Xio_studyset_dir*) realloc (this->studyset_dir, 
	(this->num_studyset_dir+1) * sizeof (Xio_studyset_dir));
    xsd = &this->studyset_dir[this->num_studyset_dir];
    this->num_studyset_dir ++;

    strncpy (xsd->path, studyset_path.c_str(), _MAX_PATH);
}

void
Xio_patient::add_plan_dir (
    std::string plan_path
)
{
    Xio_plan_dir *xtpd;

    this->plan_dir = (Xio_plan_dir*) realloc (this->plan_dir, 
	(this->num_plan_dir+1) * sizeof (Xio_plan_dir));
    xtpd = &this->plan_dir[this->num_plan_dir];
    this->num_plan_dir ++;

    strncpy (xtpd->path, plan_path.c_str(), _MAX_PATH);
}

void
Xio_patient::analyze ()
{
    itksys::Directory itk_dir;
    std::string plan_path = std::string(this->m_path) + "/plan";
    std::string studyset_path = std::string(this->m_path) + "/anatomy/studyset";

    if (itk_dir.Load (studyset_path.c_str())) {
	for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	    std::string curr_file = itk_dir.GetFile(i);
	    std::string curr_path = studyset_path + "/" + itk_dir.GetFile(i);

	    if (Xio_dir::is_xio_studyset_dir (curr_path)) {
		printf ("Adding xsd: %s\n", curr_path.c_str());
		this->add_studyset_dir (curr_path);
	    }
	}
    }

    if (itk_dir.Load (plan_path.c_str())) {
	for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	    std::string curr_file = itk_dir.GetFile(i);
	    std::string curr_path = plan_path + "/" + itk_dir.GetFile(i);

	    if (Xio_dir::is_xio_plan_dir (curr_path)) {
		printf ("Adding xtpd: %s\n", curr_path.c_str());
		this->add_plan_dir (curr_path);
	    }
	}
    }
}
