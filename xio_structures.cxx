/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <vector>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"
#include "bstrlib.h"
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"

#include "gdcm_series.h"
#include "math_util.h"
#include "plm_image_patient_position.h"
#include "plm_path.h"
#include "print_and_exit.h"
#include "rtss_polyline_set.h"
#include "xio_ct.h"
#include "xio_io.h"
#include "xio_structures.h"

/* Gdcm has a broken header file gdcmCommon.h, which defines C99 types 
   (e.g. int32_t) when missing on MSVC.  However, it does so in an incorrect 
   way that conflicts with plm_int.h (which also fixes missing C99 types).  
   The workaround is to separately define the functions in file_util.h 
   that we need. */
extern "C"
gpuit_EXPORT
void make_directory_recursive (const char *dirname);


static void
add_cms_contournames (Rtss_polyline_set *rtss, const char *filename)
{
    FILE *fp;
    struct bStream * bs;
    bstring version = bfromcstr ("");
    bstring line1 = bfromcstr ("");
    bstring line2 = bfromcstr ("");
    int skip_lines = 2;

    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    bs = bsopen ((bNread) fread, fp);

    /* Read version number */
    bsreadln (version, bs, '\n');
    btrimws (version);
    if (!strcmp ((const char*) version->data, "00061027")) {
	printf ("Version 00061027 found.\n");
	skip_lines = 5;
    }

    /* Skip line */
    bsreadln (line1, bs, '\n');

    while (1)
    {
	int rc, structure_id;

	/* Get structure name */
	rc = bsreadln (line1, bs, '\n');
	if (rc == BSTR_ERR) {
	    break;
	}
	btrimws (line1);

	/* Get structure number */
	rc = bsreadln (line2, bs, '\n');
	if (rc == BSTR_ERR) {
	    break;
	}
	btrimws (line2);
	rc = sscanf ((char*) line2->data, "%d,", &structure_id);

	if (rc != 1) {
	    if (!strcmp ((const char*) version->data, "00061027")) {
		/* This XiO version seems to write corrupted contourfiles 
		   when editing files created with previous versions. 
		   We'll assume that everything went ok.  */
		break;
	    }
	    /* GCS 2010-12-27: It's not just that version which does this.
	       What happens it that XiO leaves garbage at the end of the 
	       file.  The better way to handle this is probably to count 
	       the number of structures and then stop. */
#if defined (commentout)
	    print_and_exit ("Error parsing contournames: "
			    "contour id not found (%s)\n", line2->data);
#endif
	}

	/* Xio structures can be zero.  This is possibly not tolerated 
	   by dicom.  So we modify before inserting into the cxt. */
	structure_id ++;
	if (structure_id <= 0) {
	    print_and_exit ("Error, structure_id was less than zero\n");
	}

	/* Add structure */
	rtss->add_structure (CBString ((const char*) line1->data), 
	    CBString(), structure_id);

	/* Skip extra lines */
	for (int i = 0; i < skip_lines; i++) {
	    bsreadln (line1, bs, '\n');
	}
    }

    bdestroy (version);
    bdestroy (line1);
    bdestroy (line2);
    bsclose (bs);
    fclose (fp);
}

static void
add_cms_structure (Rtss_polyline_set *rtss, const char *filename, 
		   float z_loc)
{
    FILE *fp;
    char buf[1024];

    fp = fopen (filename, "r");
    if (!fp) {
	printf ("Error opening file %s for read\n", filename);
	exit (-1);
    }

    /* Skip first five lines */
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);

    while (1) {
	int rc;
	int structure_id, num_points;
	int point_idx, remaining_points;
	Rtss_structure *curr_structure;
	Rtss_polyline *curr_polyline;

	/* Get num points */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &num_points);
	if (rc != 1) {
	    print_and_exit ("Error parsing file %s (num_points)\n", filename);
	}

	/* Get structure number */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &structure_id);
	if (rc != 1) {
	    print_and_exit ("Error parsing file %s (structure_id)\n", 
			    filename);
	}
	
	/* Xio structures can be zero.  This is possibly not tolerated 
	   by dicom.  So we modify before inserting into the cxt. */
	structure_id ++;
	if (structure_id <= 0) {
	    print_and_exit ("Error, structure_id was less than zero\n");
	}

	/* Can this happen? */
	if (num_points == 0) {
	    break;
	}

	/* Look up the cxt structure for this id */
	curr_structure = rtss->find_structure_by_id (structure_id);
	if (!curr_structure) {
	    print_and_exit ("Couldn't reference structure with id %d\n", 
			    structure_id);
	}

	printf ("[%f %d %d]\n", z_loc, structure_id, num_points);
	curr_polyline = curr_structure->add_polyline ();
	curr_polyline->slice_no = -1;
	curr_polyline->num_vertices = num_points;
	curr_polyline->x = (float*) malloc (num_points * sizeof(float));
	curr_polyline->y = (float*) malloc (num_points * sizeof(float));
	curr_polyline->z = (float*) malloc (num_points * sizeof(float));

	point_idx = 0;
	remaining_points = num_points;
	while (remaining_points > 0) {
	    int p, line_points, line_loc;

	    fgets (buf, 1024, fp);

	    if (remaining_points > 5) {
		line_points = 5;
	    } else {
		line_points = remaining_points;
	    }
	    line_loc = 0;

	    for (p = 0; p < line_points; p++) {
		float x, y;
		int rc, this_loc;

		rc = sscanf (&buf[line_loc], "%f, %f,%n", &x, &y, &this_loc);
		if (rc != 2) {
		    print_and_exit ("Error parsing file %s (points) %s\n", 
				    filename, &buf[line_loc]);
		}

		curr_polyline->x[point_idx] = x;
		curr_polyline->y[point_idx] = -y;
		curr_polyline->z[point_idx] = z_loc;
		point_idx ++;
		line_loc += this_loc;
	    }
	    remaining_points -= line_points;
	}
    }

    fclose (fp);
}

void
xio_structures_load (
    Rtss_polyline_set *rtss, 
    char *input_dir
)
{
    const char *filename_re = "^T\\.([-\\.0-9]*)\\.WC$";

    /* Get the index file */
    std::string index_file = std::string(input_dir) + "/" + "contournames";
    if (!itksys::SystemTools::FileExists (index_file.c_str(), true)) {
	print_and_exit ("No xio contournames file found in directory %s\n", 
			input_dir);
    }

    /* Get the list of filenames */
    std::vector<std::pair<std::string,std::string> > file_names;
    xio_io_get_file_names (&file_names, input_dir, filename_re);
    if (file_names.empty ()) {
	print_and_exit ("No xio structure files found in directory %s\n", 
			input_dir);
    }

    /* Load the index file */
    rtss->init ();
    add_cms_contournames (rtss, index_file.c_str());

    /* Iterate through filenames, adding data to CXT */
    std::vector<std::pair<std::string,std::string> >::iterator it;
    it = file_names.begin();
    while (it != file_names.end()) {
	const char *filename = (*it).first.c_str();
	float z_loc = atof ((*it).second.c_str());
	printf ("File: %s, Loc: %f\n", filename, z_loc);
	add_cms_structure (rtss, filename, z_loc);
	++it;
    }

    rtss->debug ();
}

/* This is idiotic */
static void
format_xio_filename (char *fn, const char *output_dir, float z_loc)
{
    int neg;
    int z_round, z_ones, z_tenths;
    const char *neg_string;

    neg = (z_loc < 0);
    if (neg) z_loc = - z_loc;
    z_round = ROUND (z_loc * 10);
    z_ones = z_round / 10;
    z_tenths = z_round % 10;

    neg_string = neg ? "-" : "";

    if (z_ones == 0 && z_tenths == 0) {
	sprintf (fn, "%s/T.%s0.WC", output_dir, neg_string);
    } 
    else if (z_ones == 0) {
	sprintf (fn, "%s/T.%s.%d.WC", output_dir, neg_string, z_tenths);
    }
    else if (z_tenths == 0) {
	sprintf (fn, "%s/T.%s%d.WC", output_dir, neg_string, z_ones);
    }
    else {
	sprintf (fn, "%s/T.%s%d.%d.WC", output_dir, neg_string, 
	    z_ones, z_tenths);
    }
}

void
xio_structures_save (
    Rtss_polyline_set *cxt,
    Xio_ct_transform *transform,
    Xio_version xio_version,
    const char *output_dir
)
{
    FILE *fp;
    int i, j, k, z;
    char fn[_MAX_PATH];

    printf ("X_S_S: output_dir = %s\n", output_dir);

    if (!cxt->have_geometry) {
	print_and_exit ("Sorry, can't output xio format without ct geometry\n");
    }

    /* Write contournames */
    sprintf (fn, "%s/%s", output_dir, "contournames");
    make_directory_recursive (fn);
    fp = fopen (fn, "w");
    if (!fp) {
	print_and_exit ("Error opening output file %s\n", fn);
    }

    if (xio_version == XIO_VERSION_4_2_1) {
	fprintf (fp, "00031027\n");
    } else {
	fprintf (fp, "00041027\n");
    }

    fprintf (fp, "%d\n", cxt->num_structures);

    for (i = 0; i < cxt->num_structures; i++) {
	Rtss_structure *curr_structure = cxt->slist[i];
	int color = 1 + (i % 8);
	int pen = 1;
	/* Class 0 is "patient", class 1 is "Int" */
	int structure_class = (i == 0) ? 0 : 1;
	/* Name */
	fprintf (fp, "%s\n", (const char*) curr_structure->name);
	/* Structure no, density, ??, class [, date] */
	fprintf (fp, "%d,1.000000,0,%d%s\n", 
	    i+1, structure_class, 
	    (xio_version == XIO_VERSION_4_2_1) ? "" : ",19691231.190000");
	/* Grouping */
	fprintf (fp, "General\n");
	/* color, ??, pen, ??, ??, ?? */
	fprintf (fp, "%d,5,%d,1,0,0\n", color, pen);
    }
    fclose (fp);

    /* Write WC files */
    for (z = 0; z < cxt->dim[2]; z++) {
	char fn[_MAX_PATH];

	float z_offset = 0.0f;

	if ( (transform->patient_pos == PATIENT_POSITION_HFS) ||
	     (transform->patient_pos == PATIENT_POSITION_HFP) ||
	     (transform->patient_pos == PATIENT_POSITION_UNKNOWN) ) {
	    z_offset = cxt->offset[2];
	} else if ( (transform->patient_pos == PATIENT_POSITION_FFS) ||
	     (transform->patient_pos == PATIENT_POSITION_FFP) ) {
	    z_offset = - cxt->offset[2];
	}

	float z_loc = z_offset + z * cxt->spacing[2];
	format_xio_filename (fn, output_dir, z_loc);
	//sprintf (fn, "%s/T.%.1f.WC", output_dir, (ROUND (z_loc * 10) / 10.f));
	fp = fopen (fn, "w");
	if (!fp) {
	    print_and_exit ("Error opening output file %s\n", fn);
	}
	fprintf (fp, "00061013\n\n");
	fprintf (fp, "0\n0.000,0.000,0.000\n");
	/* GCS FIX: These seem to be min/max */
	fprintf (fp, "-158.1,-135.6, 147.7,  81.6\n");
	for (i = 0; i < cxt->num_structures; i++) {
	    Rtss_structure *curr_structure = cxt->slist[i];
	    for (j = 0; j < curr_structure->num_contours; j++) {
		Rtss_polyline *curr_polyline = curr_structure->pslist[j];
		if (z != curr_polyline->slice_no) {
		    continue;
		}
		fprintf (fp, "%d\n", curr_polyline->num_vertices);
		fprintf (fp, "%d\n", i+1);
		for (k = 0; k < curr_polyline->num_vertices; k++) {
		    fprintf (fp, "%6.1f,%6.1f", 
			curr_polyline->x[k] * transform->direction_cosines[0]
			    - transform->x_offset,
			curr_polyline->y[k] * transform->direction_cosines[4]
			    - transform->y_offset);
		    if ((k+1) % 5 == 0) {
			fprintf (fp, "\n");
		    }
		    else if (k < curr_polyline->num_vertices - 1) {
			fprintf (fp, ",");
		    }
		    else {
			fprintf (fp, "\n");
		    }
		}
	    }
	}
	fprintf (fp, "0\n0\n0\nBart\n");
	fclose (fp);
    }
}

void
xio_structures_apply_transform (Rtss_polyline_set *rtss, Xio_ct_transform *transform)
{
    int i, j, k;

    /* Set offsets */
    rtss->offset[0] = (rtss->offset[0] * transform->direction_cosines[0])
	+ transform->x_offset;
    rtss->offset[1] = (rtss->offset[1] * transform->direction_cosines[4])
	+ transform->y_offset;

    /* Transform structures */
    for (i = 0; i < rtss->num_structures; i++) {
	Rtss_structure *curr_structure = rtss->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Rtss_polyline *curr_polyline = curr_structure->pslist[j];
	    for (k = 0; k < curr_polyline->num_vertices; k++) {
		curr_polyline->x[k] =
		    (curr_polyline->x[k] * transform->direction_cosines[0])
		    + transform->x_offset;
		curr_polyline->y[k] =
		    (curr_polyline->y[k] * transform->direction_cosines[4])
		    + transform->y_offset;
	    }
	}
    }

}
