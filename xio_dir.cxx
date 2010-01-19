 /* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "bstrlib.h"

#include "file_util.h"
#include "plm_config.h"
#include "print_and_exit.h"
#include "xio_dir.h"

/* -----------------------------------------------------------------------
   Private functions
   ----------------------------------------------------------------------- */
static int
is_xio_patient_dir (std::string dir)
{
    itksys::Directory itk_dir;
    if (!itk_dir.Load (dir.c_str())) {
	return 0;
    }

    /* A patient directory has either an anatomy or a plan directory */
    for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	std::string curr_file = itk_dir.GetFile(i);
	std::string curr_path = dir + "/" + itk_dir.GetFile(i);
	
	if (curr_file == "anatomy") return 1;
	if (curr_file == "plan")  return 1;
    }
    return 0;
}

static int
is_xio_studyset_dir (std::string dir)
{
    itksys::Directory itk_dir;
    if (!itk_dir.Load (dir.c_str())) {
	return 0;
    }

    /* A studyset directory has either a *.CT file or a *.WC file */
    for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	std::string curr_file = itk_dir.GetFile(i);
	std::string curr_path = dir + "/" + itk_dir.GetFile(i);

	if (itksys::SystemTools::FileIsDirectory (curr_path.c_str())) {
	    continue;
	}

	if (extension_is (curr_file.c_str(), ".WC")) return 1;
	if (extension_is (curr_file.c_str(), ".CT")) return 1;
    }
    return 0;
}

static Xio_patient_dir*
xio_dir_add_patient_dir (Xio_dir *xd, std::string dir)
{
    Xio_patient_dir *xpd;
    xd->patient_dir = (Xio_patient_dir*) malloc (
	(xd->num_patient_dir+1) * sizeof (Xio_patient_dir));
    xpd = &xd->patient_dir[xd->num_patient_dir];
    xd->num_patient_dir ++;

    strncpy (xpd->path, dir.c_str(), _MAX_PATH);
    xpd->num_studyset_dir = 0;
    xpd->studyset_dir = 0;
    
    return xpd;
}

static void
xio_patient_dir_add_studyset_dir (
    Xio_patient_dir *xpd, 
    std::string studyset_path
)
{
    Xio_studyset_dir *xsd;

    xpd->studyset_dir = (Xio_studyset_dir*) malloc (
	(xpd->num_studyset_dir+1) * sizeof (Xio_studyset_dir));
    xsd = &xpd->studyset_dir[xpd->num_studyset_dir];
    xpd->num_studyset_dir ++;

    strncpy (xsd->path, studyset_path.c_str(), _MAX_PATH);
}

static void
xio_patient_dir_analyze (Xio_patient_dir *xpd)
{
    itksys::Directory itk_dir;
    std::string studyset_path = std::string(xpd->path) + "/anatomy/studyset";
    
    if (!itk_dir.Load (studyset_path.c_str())) {
	return;
    }
    for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	std::string curr_file = itk_dir.GetFile(i);
	std::string curr_path = studyset_path + "/" + itk_dir.GetFile(i);
	
	if (is_xio_studyset_dir (curr_path)) {
	    xio_patient_dir_add_studyset_dir (xpd, curr_path);
	}
    }
}

static void
xio_dir_analyze_recursive (Xio_dir *xd, std::string dir)
{
    itksys::Directory itk_dir;

    if (!itk_dir.Load (dir.c_str())) {
	return;
    }

    /* Look for top-level patient directory */
    if (is_xio_patient_dir (dir)) {
	Xio_patient_dir *curr_patient_dir 
	    = xio_dir_add_patient_dir (xd, dir);
	curr_patient_dir->type = XPD_TOPLEVEL_PATIENT_DIR;

	xio_patient_dir_analyze (curr_patient_dir);
	return;
    }

#if defined (commentout)
    /* Look for studyset directories.  
       GCS FIX: Each studyset counts as a separate patient 
       GCS FIX: Look for plan directories too. */
    else if (is_xio_studyset_dir (dir)) {
	Xio_patient_dir *curr_patient_dir 
	    = xio_dir_add_patient_dir (xpd, dir);
	curr_patient_dir->type = XPD_STUDYSET_DIR;
	
	return;
    }
#endif

    for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	std::string curr_file = itk_dir.GetFile(i);
	std::string curr_path = dir + "/" + itk_dir.GetFile(i);
	
	if (curr_file == "." || curr_file == "..") continue;

	if (itksys::SystemTools::FileIsDirectory (curr_path.c_str())) {
	    xio_dir_analyze_recursive (xd, curr_path);
	}
    }
}

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
Xio_dir*
xio_dir_create (char *input_dir)
{
    Xio_dir *xd;
    xd = (Xio_dir*) malloc (sizeof (Xio_dir));

    strncpy (xd->path, input_dir, _MAX_PATH);
    xd->num_patient_dir = -1;
    xd->patient_dir = 0;

    xio_dir_analyze (xd);
    return xd;
}

void
xio_dir_analyze (Xio_dir *xd)
{
    xd->num_patient_dir = 0;
    if (!is_directory (xd->path)) {
	return;
    }

    xio_dir_analyze_recursive (xd, std::string (xd->path));
}

int
xio_dir_num_patients (Xio_dir* xd)
{
    return xd->num_patient_dir;
}

void
xio_dir_destroy (Xio_dir* xd)
{
    if (xd->patient_dir) {
	free (xd->patient_dir);
    }
    free (xd);
}
