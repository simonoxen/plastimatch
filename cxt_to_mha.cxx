/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "volume.h"
#include "readmha.h"

#if defined (WIN32)
#include <direct.h>
#define snprintf _snprintf
#define mkdir(a, b) _mkdir (a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "render_polyline.h"
#include "getopt.h"

#define BUFLEN 2048
#define BUF (128 * 1024)

typedef struct program_parms Program_Parms;
struct program_parms {
    char* file_txt;
    char* outdir;
    char* fn_prefix;
};

typedef struct polyline POLYLINE;
struct polyline {
    int slice_no;
    int num_vertices;
    float* x;
    float* y;
    float* z;
};

typedef struct structure STRUCTURE;
struct structure {
    char name[BUFLEN];
    int num_contours;
    POLYLINE* pslist;
};
typedef struct structure_list STRUCTURE_List;
struct structure_list {
    int dim[3];
    float spacing[3];
    float offset[3];
    int num_structures;
    STRUCTURE* slist;
};
void
print_usage (void)
{
    printf ("Usage: cxt_to_mha cxt_file prefix\n");
    printf ("  The cxt_file is an ASCII file with the contours\n");
    printf ("  The prefix is (e.g.) a 4 digit patient number.\n");
    exit (-1);
}

void
load_structures (Program_Parms* parms, STRUCTURE_List* structures)
{
    FILE* fp;
    STRUCTURE* curr_structure = (STRUCTURE*) malloc (sizeof(STRUCTURE));
    POLYLINE* curr_contour = (POLYLINE*) malloc (sizeof(POLYLINE));

    float val_x = 0;
    float val_y = 0;
    float val_z = 0;

    int struct_no = 0;
    int num_pt = 0;
    int old_struct_no = -1;
    int contour_no = 0;
    int slice_idx = -1;
    char tag[BUFLEN];

    float x = 0;
    float y = 0;
    float z = 0;

    memset (curr_structure, 0, sizeof(STRUCTURE));
    memset (curr_contour, 0, sizeof(POLYLINE));
    curr_structure->num_contours = 0;
    curr_contour->num_vertices = 0;

    fp = fopen (parms->file_txt, "r");

    if (!fp) {
        printf ("Could not open contour file\n");
        exit (-1);
    }

    printf ("Loading...\n");
    while (1) {
        char buf[BUFLEN];
        char *p;

        p = fgets (buf, BUFLEN, fp);
        if (!p) {
            fprintf (stderr, "ERROR: Your file is not formatted correctly!\n");
            exit (-1);
        }
        if (!strncmp (buf, "ROI_NAMES", strlen ("ROI_NAMES"))) {
            break;
        }
        if (4 == sscanf (buf, "%s %f %f %f", tag, &val_x, &val_y, &val_z)) {
            if (strcmp ("OFFSET", tag) == 0) {
                structures->offset[0] = val_x;
                structures->offset[1] = val_y;
                structures->offset[2] = val_z;
                //printf("%s\n",tag);
            } else if (strcmp ("DIMENSION", tag) == 0) {
                structures->dim[0] = val_x;
                structures->dim[1] = val_y;
                structures->dim[2] = val_z;
                //printf("%s\n",tag);
            } else if (strcmp ("SPACING", tag) == 0) {
                structures->spacing[0] = val_x;
                structures->spacing[1] = val_y;
                structures->spacing[2] = val_z;
                //printf("%s\n",tag);
            }
        }
    }
    while (1) {
        char color[BUFLEN];
        char name[BUFLEN];
        char buf[BUFLEN];
        char *p;
        int rc;

        p = fgets (buf, BUFLEN, fp);
        if (!p) {
            fprintf (stderr, "ERROR: Your file is not formatted correctly!\n");
            exit (-1);
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_no, color, name);
        if (rc != 3) {
            break;
        }

        structures->num_structures++;
        structures->slist = (STRUCTURE*) realloc (structures->slist,
                                                  structures->num_structures * sizeof(STRUCTURE));
        curr_structure = &structures->slist[structures->num_structures - 1];
        strcpy (curr_structure->name, name);
        curr_structure->num_contours = 0;
        curr_structure->pslist = 0;
        printf ("STRUCTURE: %s\n", curr_structure->name);
    }

    while (1) {
        if (1 != fscanf (fp, "%d", &struct_no)) {
            break;
        }
        fgetc (fp);

        /* Skip contour thickness */
        while (fgetc (fp) != '|') ;

        if (1 != fscanf (fp, "%d", &num_pt)) {
            break;
        }
        fgetc (fp);

        if (1 != fscanf (fp, "%d", &slice_idx)) {
            break;
        }
        fgetc (fp);

        /* Skip uid */
        while (fgetc (fp) != '|') ;

        //printf ("%d %d %d\n", struct_no, num_pt, slice_idx);

        if (struct_no != old_struct_no) {
            old_struct_no = struct_no;
            contour_no = 0;
        }
        curr_structure = &structures->slist[struct_no - 1];
        //printf ("Gonna realloc %p, %d\n", curr_structure->pslist, contour_no);
        curr_structure->pslist = (POLYLINE*) realloc (curr_structure->pslist,
                                                      (contour_no + 1) * sizeof(POLYLINE));
        //printf ("Gonna dereference pslist\n");
        curr_contour = &curr_structure->pslist[contour_no];
        curr_contour->num_vertices = num_pt;
        curr_contour->slice_no = slice_idx;
        contour_no++;
        curr_structure->num_contours = contour_no;

        //printf ("Gonna dereference curr_contour->x\n");
        curr_contour->x = (float*) malloc (num_pt * sizeof(float));
        curr_contour->y = (float*) malloc (num_pt * sizeof(float));
        curr_contour->z = (float*) malloc (num_pt * sizeof(float));
        if (curr_contour->y == 0 || curr_contour->x == 0) {
            fprintf (stderr, "Error allocating memory");
            exit (-1);
        }
        for (int k = 0; k < num_pt; k++) {
            //printf (" --> %5d\n", k);
            if (fscanf (fp, "%f\\%f\\%f", &x, &y, &z) != 3) {
                if (fscanf (fp, "\\%f\\%f\\%f", &x, &y, &z) != 3) {
                    break;
                }
            }
            curr_contour->x[k] = x;
            curr_contour->y[k] = y;
            curr_contour->z[k] = z;
            x = 0;
            y = 0;
            z = 0;
        }
        slice_idx = 0;
        num_pt = 0;
    }
    printf ("successful!\n");
    fclose (fp);
}

int
main (int argc, char* argv[])
{
    if (argc < 3) {
        print_usage ();
        exit (0);
    }

    Program_Parms* parms = (Program_Parms*) malloc (sizeof(Program_Parms));
    STRUCTURE_List* structures = (STRUCTURE_List*) malloc (sizeof(STRUCTURE_List));
    STRUCTURE* curr_structure = (STRUCTURE*) malloc (sizeof(STRUCTURE));
    POLYLINE* curr_contour = (POLYLINE*) malloc (sizeof(POLYLINE));
    Volume* vol;

    unsigned char* img;
    unsigned char* acc_img;
    int dim[2];
    float offset[2];
    float spacing[2];
    int slice_voxels = 0;

    memset (structures, 0, sizeof(STRUCTURE_List));
    structures->num_structures = 0;
    memset (curr_structure, 0, sizeof(STRUCTURE));
    memset (curr_contour, 0, sizeof(POLYLINE));
    curr_structure->num_contours = 0;
    curr_contour->num_vertices = 0;

    parms->file_txt = argv[1];
    parms->fn_prefix = argv[2];

    load_structures (parms, structures);

    dim[0] = structures->dim[0];
    dim[1] = structures->dim[1];
    offset[0] = structures->offset[0];
    offset[1] = structures->offset[1];
    spacing[0] = structures->spacing[0];
    spacing[1] = structures->spacing[1];
    slice_voxels = dim[0] * dim[1];
    acc_img = (unsigned char*) malloc (slice_voxels * sizeof(unsigned char));
    vol = volume_create (structures->dim, structures->offset, 
			 structures->spacing, PT_UCHAR, 0, 0);
    if (vol == 0) {
        fprintf (stderr, "ERROR: failed in allocating the volume");
    }
    img = (unsigned char*) vol->img;
    for (int j = 0; j < structures->num_structures; j++) {
        curr_structure = &structures->slist[j];
        char fn[BUFLEN] = "";
        strcat (fn, parms->fn_prefix);
        strcat (fn, "_");
        strcat (fn, curr_structure->name);
        strcat (fn, ".mha");
        printf ("output filename: %s\n", fn);
        //system("PAUSE");
        memset (img, 0, structures->dim[0] * structures->dim[1] 
		* structures->dim[2] * sizeof(unsigned char));
        printf ("Writing image %s, num_contours=%d\n", fn, 
		curr_structure->num_contours);
        for (int i = 0; i < curr_structure->num_contours; i++) {
            curr_contour = &curr_structure->pslist[i];
            unsigned char* slice_img = &img[curr_contour->slice_no * dim[0] * dim[1]];
            //printf ("Slice# %3d\n", curr_contour->slice_no);
            memset (acc_img, 0, dim[0] * dim[1] * sizeof(unsigned char));
            render_slice_polyline (acc_img, dim, spacing, offset,
                                   curr_contour->num_vertices, 
				   curr_contour->x, curr_contour->y);
            for (int k = 0; k < slice_voxels; k++) {
                slice_img[k] ^= acc_img[k];
            }
        }
        write_mha (fn, vol);
        //break;
    }
    volume_free (vol);
    free (parms);
}
