/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkImageRegionIterator.h"
#include "itkRegularExpressionSeriesFileNames.h"

#include "autolabel_trainer.h"
#include "dlib_trainer.h"
#include "file_util.h"
#include "itk_image.h"
#include "plm_image.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "thumbnail.h"

Autolabel_trainer::Autolabel_trainer ()
{
    m_dt_tsv1 = 0;
    m_dt_tsv2_x = 0;
    m_dt_tsv2_y = 0;
}

Autolabel_trainer::~Autolabel_trainer ()
{
    if (m_dt_tsv1) delete m_dt_tsv1;
    if (m_dt_tsv2_x) delete m_dt_tsv2_x;
    if (m_dt_tsv2_y) delete m_dt_tsv2_y;
}

void
Autolabel_trainer::set_input_dir (const char* input_dir)
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

static std::map<float, Point> 
load_tspine_map (const char* fcsv_fn)
{
    Labeled_pointset ps;
    ps.load_fcsv (fcsv_fn);

    /* Generate map from t-spine # to (x,y,z) point */
    std::map<float, Point> t_map;
    for (unsigned int i = 0; i < ps.point_list.size(); i++) {
	if (ps.point_list[i].label == "C7") {
	    t_map.insert (std::pair<float,Point> (0, 
		    Point (ps.point_list[i].p[0],
			ps.point_list[i].p[1],
			ps.point_list[i].p[2])));
	}
	else if (ps.point_list[i].label == "L1") {
	    t_map.insert (std::pair<float,Point> (13, 
		    Point (ps.point_list[i].p[0],
			ps.point_list[i].p[1],
			ps.point_list[i].p[2])));
	}
	else {
	    float t;
	    int rc = sscanf (ps.point_list[i].label.c_str(), "T%f", &t);
	    if (rc != 1) {
		/* Not a vertebra point */
		continue;
	    }
	    if (t > 0.25 && t < 12.75) {
		t_map.insert (std::pair<float,Point> (t, 
			Point (ps.point_list[i].p[0],
			    ps.point_list[i].p[1],
			    ps.point_list[i].p[2])));
	    }
	}
    }
    return t_map;
}

void
Autolabel_trainer::load_input_file (
    const char* nrrd_fn,
    const char* fcsv_fn)
{
    printf ("Loading %s\nLoading %s\n---\n", nrrd_fn, fcsv_fn);

    /* Create map from spine # to point from fcsv file */
    std::map<float,Point> t_map = load_tspine_map (fcsv_fn);

    /* If we want to use interpolation, we need to sort, and make 
       a "back-map" from z-pos to t-spine */

    /* Otherwise, for the simple case, we're good to go. 
       Load the input image. */
    Plm_image *pli;
    pli = plm_image_load (nrrd_fn, PLM_IMG_TYPE_ITK_FLOAT);
    Thumbnail thumb;
    thumb.set_input_image (pli);
    thumb.set_thumbnail_dim (16);
    thumb.set_thumbnail_spacing (25.0f);

    /* Get the samples and labels */
    std::map<float, Point>::iterator it;
    for (it = t_map.begin(); it != t_map.end(); ++it) {
	thumb.set_slice_loc (it->second.p[2]);
	FloatImageType::Pointer thumb_img = thumb.make_thumbnail ();
	itk::ImageRegionIterator< FloatImageType > thumb_it (
	    thumb_img, thumb_img->GetLargestPossibleRegion());
	Dlib_trainer::Dense_sample_type d;
	for (int j = 0; j < 256; j++) {
	    d(j) = thumb_it.Get();
	    ++thumb_it;
	}

        if (this->m_dt_tsv1) {
            this->m_dt_tsv1->m_samples.push_back (d);
            this->m_dt_tsv1->m_labels.push_back (it->first);
        }
        if (this->m_dt_tsv2_x) {
            this->m_dt_tsv2_x->m_samples.push_back (d);
            this->m_dt_tsv2_x->m_labels.push_back (it->second.p[0]);
        }
        if (this->m_dt_tsv2_y) {
            this->m_dt_tsv2_y->m_samples.push_back (d);
            this->m_dt_tsv2_y->m_labels.push_back (it->second.p[1]);
        }
    }

    delete pli;
}

void
Autolabel_trainer::load_inputs ()
{
    if (m_task == "" || m_input_dir == "") {
	print_and_exit ("Error: inputs not fully specified.\n");
    }

    if (m_task == "la") {
        /* Not yet implemented */
    }
    else if (m_task == "tsv1") {
	m_dt_tsv1 = new Dlib_trainer;
    }
    else if (m_task == "tsv2") {
	m_dt_tsv2_x = new Dlib_trainer;
	m_dt_tsv2_y = new Dlib_trainer;
    }
    else {
	print_and_exit ("Error: unsupported autolabel-train task (%s)\n",
	    m_task.c_str());
    }

    load_input_dir_recursive (m_input_dir);
}

void
Autolabel_trainer::set_task (const char* task)
{
    m_task = task;
}

void
Autolabel_trainer::train ()
{
    if (this->m_dt_tsv1) {
        Pstring output_net_fn;
        output_net_fn.format ("%s/tsv1.net", m_output_dir.c_str());
        m_dt_tsv1->set_krr_gamma (-9, -5, 0.5);
        m_dt_tsv1->train_krr ();
        m_dt_tsv1->save_net (output_net_fn);
    }
    if (this->m_dt_tsv2_x) {
        Pstring output_net_fn;
        output_net_fn.format ("%s/tsv2_x.net", m_output_dir.c_str());
        m_dt_tsv2_x->set_krr_gamma (-9, -5, 0.5);
        m_dt_tsv2_x->train_krr ();
        m_dt_tsv2_x->save_net (output_net_fn);
    }
    if (this->m_dt_tsv2_y) {
        Pstring output_net_fn;
        output_net_fn.format ("%s/tsv2_y.net", m_output_dir.c_str());
        m_dt_tsv2_y->set_krr_gamma (-9, -5, 0.5);
        m_dt_tsv2_y->train_krr ();
        m_dt_tsv2_y->save_net (output_net_fn);
    }
}

void
Autolabel_trainer::save_csv ()
{
    if (this->m_dt_tsv1) {
        Pstring output_csv_fn;
        output_csv_fn.format ("%s/tsv1.csv", m_output_dir.c_str());
        this->m_dt_tsv1->save_csv (output_csv_fn);
    }
    if (this->m_dt_tsv2_x) {
        Pstring output_csv_fn;
        output_csv_fn.format ("%s/tsv2_x.csv", m_output_dir.c_str());
        this->m_dt_tsv2_x->save_csv (output_csv_fn);
    }
    if (this->m_dt_tsv2_y) {
        Pstring output_csv_fn;
        output_csv_fn.format ("%s/tsv2_y.csv", m_output_dir.c_str());
        this->m_dt_tsv2_y->save_csv (output_csv_fn);
    }
}

/* tsacc = testset accuracy */
void
Autolabel_trainer::save_tsacc ()
{
    /* Save the output files */
    if (this->m_dt_tsv1) {
        Pstring output_tsacc_fn;
        output_tsacc_fn.format ("%s/tsv1_tsacc.txt", m_output_dir.c_str());
        this->m_dt_tsv1->save_tsacc (output_tsacc_fn);
    }
    if (this->m_dt_tsv2_x) {
        Pstring output_tsacc_fn;
        output_tsacc_fn.format ("%s/tsv2_x_tsacc.txt", m_output_dir.c_str());
        this->m_dt_tsv2_x->save_tsacc (output_tsacc_fn);
    }
    if (this->m_dt_tsv2_y) {
        Pstring output_tsacc_fn;
        output_tsacc_fn.format ("%s/tsv2_y_tsacc.txt", m_output_dir.c_str());
        this->m_dt_tsv2_y->save_tsacc (output_tsacc_fn);
    }
}

void
autolabel_train (Autolabel_train_parms *parms)
{
    Autolabel_trainer at;

    at.set_input_dir ((const char*) parms->input_dir);
    at.m_output_dir = parms->output_dir;
    at.set_task ((const char*) parms->task);
    at.load_inputs ();
    at.train ();
    at.save_csv ();
    at.save_tsacc ();
}
