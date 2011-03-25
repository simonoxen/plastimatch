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
#include "xio_plan.h"

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
	
	if (itksys::SystemTools::FileIsDirectory (curr_path.c_str())) {
	    if (curr_file == "anatomy") return 1;
	    if (curr_file == "plan")  return 1;
	}
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

static int
is_xio_plan_dir (std::string dir)
{
    itksys::Directory itk_dir;
    if (!itk_dir.Load (dir.c_str())) {
	return 0;
    }

    /* A plan directory has a plan file */
    for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	std::string curr_file = itk_dir.GetFile(i);
	std::string curr_path = dir + "/" + itk_dir.GetFile(i);

	if (itksys::SystemTools::FileIsDirectory (curr_path.c_str())) {
	    continue;
	}

	if (curr_file == "plan") return 1;
    }
    return 0;
}

static Xio_patient_dir*
xio_dir_add_patient_dir (Xio_dir *xd, std::string dir)
{
    Xio_patient_dir *xpd;
    xd->patient_dir = (Xio_patient_dir*) realloc (xd->patient_dir, 
	(xd->num_patient_dir+1) * sizeof (Xio_patient_dir));
    xpd = &xd->patient_dir[xd->num_patient_dir];
    xd->num_patient_dir ++;

    strncpy (xpd->path, dir.c_str(), _MAX_PATH);
    xpd->num_studyset_dir = 0;
    xpd->num_plan_dir = 0;
    xpd->studyset_dir = 0;
    xpd->plan_dir = 0;

    return xpd;
}

static void
xio_patient_dir_add_studyset_dir (
    Xio_patient_dir *xpd, 
    std::string studyset_path
)
{
    Xio_studyset_dir *xsd;

    xpd->studyset_dir = (Xio_studyset_dir*) realloc (xpd->studyset_dir, 
	(xpd->num_studyset_dir+1) * sizeof (Xio_studyset_dir));
    xsd = &xpd->studyset_dir[xpd->num_studyset_dir];
    xpd->num_studyset_dir ++;

    strncpy (xsd->path, studyset_path.c_str(), _MAX_PATH);
}

static void
xio_patient_dir_add_plan_dir (
    Xio_patient_dir *xpd,
    std::string plan_path
)
{
    Xio_plan_dir *xtpd;

    xpd->plan_dir = (Xio_plan_dir*) realloc (xpd->plan_dir, 
	(xpd->num_plan_dir+1) * sizeof (Xio_plan_dir));
    xtpd = &xpd->plan_dir[xpd->num_plan_dir];
    xpd->num_plan_dir ++;

    strncpy (xtpd->path, plan_path.c_str(), _MAX_PATH);
}

static void
xio_patient_dir_analyze (Xio_patient_dir *xpd)
{
    itksys::Directory itk_dir;
    std::string plan_path = std::string(xpd->path) + "/plan";
    std::string studyset_path = std::string(xpd->path) + "/anatomy/studyset";

    if (itk_dir.Load (studyset_path.c_str())) {
	for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	    std::string curr_file = itk_dir.GetFile(i);
	    std::string curr_path = studyset_path + "/" + itk_dir.GetFile(i);

	    if (is_xio_studyset_dir (curr_path)) {
		printf ("Adding xsd: %s\n", curr_path.c_str());
		xio_patient_dir_add_studyset_dir (xpd, curr_path);
	    }
	}
    }

    if (itk_dir.Load (plan_path.c_str())) {
	for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++) {
	    std::string curr_file = itk_dir.GetFile(i);
	    std::string curr_path = plan_path + "/" + itk_dir.GetFile(i);

	    if (is_xio_plan_dir (curr_path)) {
		printf ("Adding xtpd: %s\n", curr_path.c_str());
		xio_patient_dir_add_plan_dir (xpd, curr_path);
	    }
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
	xio_patient_dir_analyze (curr_patient_dir);
	return;
    }

    /* Look for plan directories.
       GCS FIX: Each plan counts as a separate patient */
    else if (is_xio_plan_dir (dir)) {

	Xio_patient_dir *xpd
	    = xio_dir_add_patient_dir (xd, dir);
	xio_patient_dir_add_plan_dir (xpd, dir);
	return;
    }

    /* Look for studyset directories.  
       GCS FIX: Each studyset counts as a separate patient */
    else if (is_xio_studyset_dir (dir)) {
	Xio_patient_dir *xpd
	    = xio_dir_add_patient_dir (xd, dir);
	xio_patient_dir_add_studyset_dir (xpd, dir);
	return;
    }

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
#if defined (commentout)
Xio_dir*
xio_dir_create (const char *input_dir)
{
    Xio_dir *xd;
    xd = (Xio_dir*) malloc (sizeof (Xio_dir));

    xd->path = input_dir;
    xd->num_patient_dir = -1;
    xd->patient_dir = 0;

    xio_dir_analyze (xd);
    return xd;
}

void
xio_dir_destroy (Xio_dir* xd)
{
    if (xd->patient_dir) {
	free (xd->patient_dir);
    }
    free (xd);
}
#endif

Xio_dir::Xio_dir (const char *input_dir)
{
    this->path = input_dir;
    this->num_patient_dir = -1;
    this->patient_dir = 0;

    xio_dir_analyze (this);
}

Xio_dir::~Xio_dir ()
{
    if (this->patient_dir) {
	free (this->patient_dir);
    }
}

void
xio_dir_analyze (Xio_dir *xd)
{
    xd->num_patient_dir = 0;
    if (!is_directory ((const char*) xd->path)) {
	return;
    }

    xio_dir_analyze_recursive (xd, std::string ((const char*) xd->path));
}

int
xio_dir_num_patients (Xio_dir* xd)
{
    return xd->num_patient_dir;
}

Xio_studyset_dir*
xio_plan_dir_get_studyset_dir (Xio_plan_dir* xtpd)
{
    char studyset[_MAX_PATH];
    Xio_studyset_dir *xsd;
    char *plan_dir;
    char *patient_dir;

    xsd = (Xio_studyset_dir*) malloc (sizeof (Xio_studyset_dir));
    
    if (is_xio_plan_dir (xtpd->path)) {
	/* Get studyset name from plan */
	std::string plan_file = std::string(xtpd->path) + "/plan";
	printf ("plan_file: %s\n", plan_file.c_str());
	xio_plan_get_studyset (plan_file.c_str(), studyset);

	/* Obtain patient directory from plan directory */
	plan_dir = file_util_parent (xtpd->path);
	patient_dir = file_util_parent (plan_dir);
	printf ("plan_dir: %s\n", plan_file.c_str());
	printf ("patient_dir: %s\n", patient_dir);

	/* Set studyset directory */
	std::string studyset_path = std::string(patient_dir) +
	    "/anatomy/studyset/" + std::string(studyset);
	strncpy(xsd->path, studyset_path.c_str(), _MAX_PATH);

	free (plan_dir);
	free (patient_dir);
    }

    return xsd;
}
