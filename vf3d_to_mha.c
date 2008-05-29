/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Convert's Vlad's vf3d vector field format to ITK mha format */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

#define BUFLEN 1024

char header_pat[] = 
    "ObjectType = Image\n"
    "NDims = 3\n"
    "BinaryData = True\n"
    "BinaryDataByteOrderMSB = False\n"
	"TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
    "Offset = %s\n"
	"CenterOfRotation = 0 0 0\n"
    "ElementSpacing = %f %f %f\n"
    "DimSize = %d %d %d\n"
	"AnatomicalOrientation = RAI\n"
	"ElementNumberOfChannels = 3\n"
    "ElementType = MET_FLOAT\n"
    "ElementDataFile = LOCAL\n"
    ;

char *os = "0. 0. 0.";

int main (int argc, char* argv[])
{
    FILE *fp1, *fp2;
    int nx, ny, nz, ntotal;
    int i, j;
    float sz[3], vv;
    char buf[BUFLEN];
    char key[]="IAMA3DVECTORFIELD";

    if (argc != 4) {
	printf ("Usage: %s infile.vf3d outfile.mha offset_string\n", argv[0]);
	exit (1);
    }
    if (!(fp1 = fopen(argv[1],"rb"))) {
	printf ("Error opening file \"%s\" for read\n", argv[1]);
	exit (1);
    }
    if (!(fp2 = fopen(argv[2],"wb"))) {
	printf ("Error opening file \"%s\" for write\n", argv[2]);
	exit (1);
    }
    os = argv[3];

    if ((!fgets (buf, BUFLEN, fp1)) || (!strcmp(buf, key))) {
	printf ("File format error 1 in %s\n", argv[1]);
	exit (1);
    }
    fgets (buf, BUFLEN, fp1);
    sscanf(buf, "%d %d %d %g %g %g", &nx, &ny, &nz, &(sz[0]), &(sz[1]), &(sz[2]));
    fprintf (fp2, header_pat, os, sz[0], sz[1], sz[2], nx, ny, nz);

    ntotal = nx * ny * nz;
    for (i=0; i<ntotal; i++) {
	for (j=0; j<3; j++) {
	    fread(&vv, 4, 1, fp1);
	    vv *= sz[j];
	    fwrite(&vv,4, 1, fp2);
	}
    }
    fclose (fp1);
    fclose (fp2);

    return 0;
}
