/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* Convert's Vlad's vox format to ITK mha format */
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

#define BUFLEN 1024

char header_pat[] = 
    "ObjectType = Image\n"
    "NDims = 3\n"
    "BinaryData = True\n"
    "BinaryDataByteOrderMSB = False\n"
    "Offset = %s\n"
    "ElementSpacing = %f %f %f\n"
    "DimSize = %d %d %d\n"
	"AnatomicalOrientation = RAI\n"
    "ElementType = MET_SHORT\n"
    "ElementDataFile = LOCAL\n"
    ;
/*
char header_pat[] = 
    "ObjectType = Image\n"
    "NDims = 3\n"
    "BinaryData = True\n"
    "BinaryDataByteOrderMSB = False\n"
    "Offset = %s\n"
    "ElementSpacing = %f %f %f\n"
    "DimSize = %d %d %d\n"
	"AnatomicalOrientation = RAI\n"
    "ElementType = MET_UCHAR\n"
    "ElementDataFile = LOCAL\n"
    ;
*/

char *os = "-250 -250 -107.5";

int main (int argc, char* argv[])
{
    FILE *fp1, *fp2;
    int nx, ny, nz;
    float sx, sy, sz;
    int i;
	unsigned char c;

    if (argc != 4) {
	printf ("Usage: %s infile outfile offset_string\n", argv[0]);
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

    for (i = 0; i < 5; i++) {
	char buf[BUFLEN];
	if (!fgets (buf, BUFLEN, fp1)) {
	    printf ("File format error 1 in %s\n", argv[1]);
	    exit (1);
	}
	switch (i) {
	    int rc;
	case 2:
	    rc = sscanf (buf, "%d %d %d", &nx, &ny, &nz);
	    if (rc != 3) {
		printf ("File format error 2 in %s (%s)\n", argv[1], buf);
		exit (1);
	    }
	    break;
	case 3:
	    rc = sscanf (buf, "%g %g %g", &sx, &sy, &sz);
	    if (rc != 3) {
		printf ("File format error 3 in %s\n", argv[1]);
		exit (1);
	    }
	    break;
	}
    }
    fprintf (fp2, header_pat, os, sx, sy, sz, nx, ny, nz);

	fread(&c,1,1,fp1);
    while (!feof(fp1)) {
		fwrite(&c,1,1,fp2);
		fread(&c,1,1,fp1);
	}

    fclose (fp1);
    fclose (fp2);
    return 0;
}
