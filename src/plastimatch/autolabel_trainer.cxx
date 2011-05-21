/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"

#include "autolabel_trainer.h"
#include "bstrlib.h"
#include "file_util.h"
#include "print_and_exit.h"

void
Autolabel_trainer::load_input_dir (const char* input_dir)
{
    if (!itksys::SystemTools::FileIsDirectory (input_dir)) {
	print_and_exit ("Error: \'%s\' is not a directory\n", input_dir);
    }

    /* We can't load the data yet, since we don't know the task */
    m_input_dir = input_dir;
}

void
Autolabel_trainer::load_input_dir_recursive (std::string input_dir)
{
    itksys::Directory itk_dir;

    if (!itk_dir.Load (input_dir.c_str())) {
	print_and_exit ("Error loading directory (%s)\n", input_dir.c_str());
    }

    for (unsigned long i = 0; i < itk_dir.GetNumberOfFiles(); i++)
    {
	/* Skip "." and ".." */
	if (!strcmp (itk_dir.GetFile(i), ".") 
	    || !strcmp (itk_dir.GetFile(i), ".."))
	{
	    continue;
	}

	/* Make fully specified filename */
	std::string fn = input_dir + "/" + itk_dir.GetFile(i);

	/* Process directories recursively */
	if (itksys::SystemTools::FileIsDirectory (fn.c_str())) {
	    load_input_dir_recursive (fn);
	}

	/* Check for .nrrd files */
	if (extension_is (fn.c_str(), "nrrd")) {
	    /* Does .nrrd file have a corresponding .fcsv file? */
	    std::string fcsv_fn = fn;
	    fcsv_fn.replace (fn.length()-4, 4, "fcsv");

	    if (file_exists (fcsv_fn.c_str())) {
		load_input_file (fn.c_str(), fcsv_fn.c_str());
	    }
	}
    }

}

void
Autolabel_trainer::load_input_file_la (
    const char* nrrd_fn,
    const char* fcsv_fn)
{
    print_and_exit ("Error: load_input_file_la not yet implemented\n");
}

void
Autolabel_trainer::load_input_file_tsv1 (
    const char* nrrd_fn,
    const char* fcsv_fn)
{
    print_and_exit ("Error: load_input_file_tsv1 not yet implemented\n");
}

void
Autolabel_trainer::load_input_file_tsv2 (
    const char* nrrd_fn,
    const char* fcsv_fn)
{
    print_and_exit ("Error: load_input_file_tsv2 not yet implemented\n");
}

void
Autolabel_trainer::load_input_file (
    const char* nrrd_fn,
    const char* fcsv_fn)
{
    printf ("Loading\n  %s\n  %s\n", nrrd_fn, fcsv_fn);
    if (m_task == "") {
	load_input_file_la (nrrd_fn, fcsv_fn);
    }
    else if (m_task == "") {
	load_input_file_tsv1 (nrrd_fn, fcsv_fn);
    }
    else if (m_task == "") {
	load_input_file_tsv2 (nrrd_fn, fcsv_fn);
    }
    else {
	print_and_exit ("Error: unsupported autolabel-train task (%s)\n",
	    m_task.c_str());
    }
}

void
Autolabel_trainer::set_task (const char* task)
{
}

void
Autolabel_trainer::save_libsvm (const char* output_libsvm_fn)
{
    if (m_task == "" || m_input_dir == "") {
	print_and_exit ("Error saving libsvm, inputs not fully specified.\n");
    }

    /* Load the data according to task specification */
    load_input_dir_recursive (m_input_dir);
}
