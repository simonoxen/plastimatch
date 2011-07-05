/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "file_util.h"
#include "math_util.h"
#include "pointset.h"
#include "print_and_exit.h"

Pointset*
pointset_create (void)
{
    Pointset *ps;
    ps = (Pointset*) malloc (sizeof (Pointset));
    memset (ps, 0, sizeof (Pointset));
    return ps;
}

void
pointset_destroy (Pointset *ps)
{
    if (ps->points) {
	free (ps->points);
    }
    free (ps);
}

void
pointset_add_point (Pointset *ps, float lm[3])
{
    ps->num_points ++;
    pointset_resize (ps, ps->num_points);

    /* Note: Slicer landmarks are in RAS coordinates. 
       Change RAS to LPS (note that LPS == ITK RAI). */
    ps->points[(ps->num_points-1)*3 + 0] = - lm[0];
    ps->points[(ps->num_points-1)*3 + 1] = - lm[1];
    ps->points[(ps->num_points-1)*3 + 2] = lm[2];
}

void
pointset_add_point_noadjust (Pointset *ps, float lm[3])
{
    ps->num_points ++;
    pointset_resize (ps, ps->num_points);

    /* Note: no RAS to LPS adjustment */
    ps->points[(ps->num_points-1)*3 + 0] = lm[0];
    ps->points[(ps->num_points-1)*3 + 1] = lm[1];
    ps->points[(ps->num_points-1)*3 + 2] = lm[2];
}


static Pointset *
pointset_load_fcsv (const char *fn)
{
    FILE *fp;
    Pointset *ps;
    char s[1024];

    fp = fopen (fn, "r");
    if (!fp) {
	return 0;
    }

    /* Check if this file is an fcsv file */
    fgets (s, 1024, fp);
    if (strncmp (s, "# Fiducial List file", strlen ("# Fiducial List file")))
    {
	fclose (fp);
	return 0;
    }

    /* Got an fcsv file.  Parse it. */
    ps = pointset_create ();
    while (!feof(fp)) {
	char *s2;
	float lm[3];
	int land_sel, land_vis;
	int rc;

        fgets (s, 1024, fp);
	if (feof(fp)) break;
        if (s[0]=='#') continue;

	// skip the label field assuming it does not contain commas
        s2 = strchr(s,',');
	
        rc = sscanf (s2, ",%f,%f,%f,%d,%d\n", 
	    &lm[0], &lm[1], &lm[2], &land_sel, &land_vis);
	if (rc != 5) {
	    print_and_exit ("Error parsing landmark file: %s\n", fn);
	}
	ps->num_points ++;
	pointset_resize (ps, ps->num_points);

	/* Note: Slicer landmarks are in RAS coordinates. 
	   Change RAS to LPS (note that LPS == ITK RAI). */
	ps->points[(ps->num_points-1)*3 + 0] = - lm[0];
	ps->points[(ps->num_points-1)*3 + 1] = - lm[1];
	ps->points[(ps->num_points-1)*3 + 2] = lm[2];
    }
    fclose (fp);

    return ps;
}

static Pointset *
pointset_load_txt (const char *fn)
{
    FILE *fp;
    Pointset *ps;
    char s[1024];

    fp = fopen (fn, "r");
    if (!fp) {
	return 0;
    }

    /* Parse as txt file */
    ps = pointset_create ();
    while (!feof(fp)) {
	float lm[3];
	int rc;

        fgets (s, 1024, fp);
	if (feof(fp)) break;
        if (s[0]=='#') continue;

        rc = sscanf (s, "%f , %f , %f\n", &lm[0], &lm[1], &lm[2]);
	if (rc != 3) {
	    rc = sscanf (s, "%f %f %f\n", &lm[0], &lm[1], &lm[2]);
	}
	if (rc != 3) {
	    print_and_exit ("Error parsing landmark file: %s\n", fn);
	}
	ps->num_points ++;
	pointset_resize (ps, ps->num_points);

	/* Assume LPS */
	ps->points[(ps->num_points-1)*3 + 0] = lm[0];
	ps->points[(ps->num_points-1)*3 + 1] = lm[1];
	ps->points[(ps->num_points-1)*3 + 2] = lm[2];
    }
    fclose (fp);

    return ps;
}

Pointset*
pointset_load (const char *fn)
{
    Pointset *ps;

    /* First try to load fcsv */
    ps = pointset_load_fcsv (fn);
    if (ps) return ps;

    /* If that doesn't work, try loading ASCII */
    ps = pointset_load_txt (fn);
    return ps;
}

static void
pointset_save_txt (Pointset* ps, const char *fn)
{
    int i;
    FILE *fp;

    fp = fopen (fn, "w");
    if (!fp) return;

    for (i = 0; i < ps->num_points; i++) {
	fprintf (fp, "%f %f %f\n", 
	    ps->points[i*3+0], 
	    ps->points[i*3+1], 
	    ps->points[i*3+2]);
    }
    fclose (fp);
}

static void
pointset_save_fcsv (Pointset* ps, const char *fn)
{
    int i;
    FILE *fp;

    fp = fopen (fn, "w");
    if (!fp) return;

    fprintf (fp, 
	"# Fiducial List file %s\n"
	"# version = 2\n"
	"# name = plastimatch-fiducials\n"
	"# numPoints = %d\n"
	"# symbolScale = 5\n"
	"# symbolType = 12\n"
	"# visibility = 1\n"
	"# textScale = 4.5\n"
	"# color = 0.4,1,1\n"
	"# selectedColor = 1,0.5,0.5\n"
	"# opacity = 1\n"
	"# ambient = 0\n"
	"# diffuse = 1\n"
	"# specular = 0\n"
	"# power = 1\n"
	"# locked = 0\n"
	"# numberingScheme = 0\n"
	"# columns = label,x,y,z,sel,vis\n",
	fn, 
	ps->num_points);

    for (i = 0; i < ps->num_points; i++) {
	fprintf (fp, "p-%03d,%f,%f,%f,1,1\n", 
	    i,
	    - ps->points[i*3+0], 
	    - ps->points[i*3+1], 
	    ps->points[i*3+2]);
    }
    fclose (fp);
}

void
pointset_save_fcsv_by_cluster (Pointset* ps, int *clust_id, int which_cluster, const char *fn)
{
    int i;
    int symbol;
    FILE *fp;
    
    // symbolType, see
    //http://www.slicer.org/slicerWiki/index.php/Modules:Fiducials-Documentation-3.4
    symbol =which_cluster+2; 
    if (symbol > 13) symbol -=13;

    fp = fopen (fn, "w");
    if (!fp) return;

    int num_points_in_cluster=0;
    for (i = 0; i < ps->num_points; i++) {
	if (clust_id[i] == which_cluster) num_points_in_cluster++;	
        }

    fprintf (fp, 
	"# Fiducial List file %s\n"
	"# version = 2\n"
	"# name = plastimatch-fiducials\n"
	"# numPoints = %d\n"
	"# symbolScale = 5\n"
	"# symbolType = %d\n"
	"# visibility = 1\n"
	"# textScale = 4.5\n"
	"# color = 0.4,1,1\n"
	"# selectedColor = 1,0.5,0.5\n"
	"# opacity = 1\n"
	"# ambient = 0\n"
	"# diffuse = 1\n"
	"# specular = 0\n"
	"# power = 1\n"
	"# locked = 0\n"
	"# numberingScheme = 0\n"
	"# columns = label,x,y,z,sel,vis\n",
	fn, 
	num_points_in_cluster,
	symbol);

    for (i = 0; i < ps->num_points; i++) {
	if (clust_id[i] == which_cluster)
	    fprintf (fp, "p-%03d,%f,%f,%f,1,1\n", 
		    i,
		    - ps->points[i*3+0], 
		    - ps->points[i*3+1], 
		    ps->points[i*3+2]);
	}
    fclose (fp);
}


void
pointset_save (Pointset* ps, const char *fn)
{
    if (extension_is (fn, "fcsv")) {
	pointset_save_fcsv (ps, fn);
    } else {
	pointset_save_txt (ps, fn);
    }
}

void
pointset_resize (Pointset *ps, int new_size)
{
    ps->num_points = new_size;
    ps->points = (float*) realloc (ps->points, 
	    3 * (ps->num_points) * sizeof(float));
}

void
pointset_debug (Pointset* ps)
{
    int i;
    printf ("Pointset:\n");
    for (i = 0; i < ps->num_points; i++) {
	printf ("  %10f %10f %10f\n", 
	    ps->points[i*3 + 0],
	    ps->points[i*3 + 1],
	    ps->points[i*3 + 2]);
    }
}

/* New pointset */
void
Pointset_new::load_fcsv (const char *fn)
{
    FILE *fp;
    char s[1024];

    fp = fopen (fn, "r");
    if (!fp) {
	return;
    }

    /* Check if this file is an fcsv file */
    fgets (s, 1024, fp);
    if (strncmp (s, "# Fiducial List file", strlen ("# Fiducial List file")))
    {
	fclose (fp);
	return;
    }

    /* Got an fcsv file.  Parse it. */
    while (!feof(fp)) {
	float lm[3];
	int land_sel, land_vis;
	int rc;

        fgets (s, 1024, fp);
	if (feof(fp)) break;
        if (s[0]=='#') continue;

	char buf[1024];
        rc = sscanf (s, "%1023[^,],%f,%f,%f,%d,%d\n", buf, 
	    &lm[0], &lm[1], &lm[2], &land_sel, &land_vis);
	if (rc != 6) {
	    print_and_exit ("Error parsing landmark file: %s "
		"(rc=%d,str=%s,buf=%s)\n", fn, rc, buf);
	}

	/* GCS FIX: Does this method of using STL containers 
	   cause a copy of lp? */
	/* Note: Slicer landmarks are in RAS coordinates. 
	   Change RAS to LPS (note that LPS == ITK RAI). */
	Labeled_point lp;
	lp.label = buf;
	lp.p[0] = - lm[0];
	lp.p[1] = - lm[1];
	lp.p[2] = lm[2];
	point_list.push_back (lp);
    }
    fclose (fp);
}
