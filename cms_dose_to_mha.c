#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "plm_int.h"


#define XIO_VERSION_450       1
#define XIO_VERSION_421       2
#define XIO_VERSION_UNKNOWN   3

/* function to perform endian swaps when going from Big-Endian
 * to little-endian or vice-versa
 */
void 
int_endian(int *arg)
{
    char lenbuf[4];
    char tmpc;
    memcpy(lenbuf,(const char *)arg,4);
    tmpc=lenbuf[0]; lenbuf[0]=lenbuf[3]; lenbuf[3]=tmpc;
    tmpc=lenbuf[1]; lenbuf[1]=lenbuf[2]; lenbuf[2]=tmpc;
    memcpy((char *)arg,lenbuf,4);
}

int
main (int argc, char *argv[])
{
    FILE *ifp; FILE *ofp;

    long offset; //deleteme when done debugging.

    char *mode = {"rc"}; /*rt vs. rc */
    char buf[1024];
    char myarray[1024];
    char *result = NULL;
    int nn; int i; int j; int k;
    //dimension sizes
    double rx; double ry; double rz;
    double ox; double oy; double oz;
    int nx; int ny; int nz;
    //element spacing
    double dx; double dy; double dz;
    //offset (top left corner of first slice)
    double topx; double topy; double topz;

    int dose;
    int found;
    int o,p,q,l,m,n; short ***data;
    char *fn;
    int xio_version;
    float xio_dose_scale;
    int rc;

    if (argc != 3) {
	printf("Usage: cms_dose_to_mha cmsdose outputfile.mha\n");
	exit(0);
    }

    l = 520; n = 520; m = 520;
    data = malloc(l*sizeof(short**));
    for(o=0;o<l;o++)
    {
	data[o] = malloc(m*sizeof(short*));
	for(p=0;p<m;p++)
	{
	    data[o][p] = malloc(n*sizeof(short));
	    for(q=0;q<n;q++)
	    {
		data[o][p][q] = 0;
	    }
	}
    }

    fn = argv[1];
    ifp = fopen (fn, mode);
    if (ifp == NULL) {
	fprintf(stderr, "Can't open input file in.list!\n");
	exit(1);
    }

    /* Get version number */
    fgets (buf, 1024, ifp);
    if (!strncmp (buf, "006d101e", strlen("006d101e"))) {
	xio_version = XIO_VERSION_450;
    } else if (!strncmp (buf, "004f101e", strlen("004f101e"))) {
	xio_version = XIO_VERSION_421;
    } else {
	xio_version = XIO_VERSION_UNKNOWN;
    }

    /* Get dose scale factor */
    found = 0;
    if (xio_version == XIO_VERSION_450 
	|| xio_version == XIO_VERSION_421) {
	float dummy;

	/* Skip line 2 */
	fgets (buf, 1024, ifp);

	/* Find dose scale line */
	for (i = 0; i < 25; i++){
	    fgets (buf, 1024, ifp);
	    rc = sscanf (buf, "%g,%g", &dummy, &xio_dose_scale);
	    if (rc == 2) {
		found = 1;
		break;
	    }
	}
	
	if (!found) {
	    printf ("Sorry, couldn't parse dose scale: %s\n", buf);
	    exit (-1);
	}
    } else {
	xio_dose_scale = 1.0;
    }

    printf ("Dose scale = %g\n", xio_dose_scale);

    /* Search for geometry string */
    found = 0;
    for (i = 0; i < 25; i++){
	fgets (buf, 1024, ifp);
	printf ("LINE: %s",buf);
	for (j = 0; j < 200; j++) {
	    myarray[j] = buf[j];
	}

	/* Check the number of commas in the line */
	nn=0;
	result = strtok (buf,",");
	while (result!=NULL) {
	    result = strtok( NULL, ",");
	    nn++;
	}

	if (nn==10) {
	    printf ("%s\n\n","found 9 commas");
	    found = 1;
	    break;
	}
    }

    if (!found) {
	printf ("Sorry, couldn't parse dose file: %s\n", fn);
	exit (-1);
    }

    printf("Target String: %s",myarray);
    result = strtok (myarray,",");
    printf("first number is: %s\n", result);

    //////////////rx,ry,and rz are the "dimension size" 
    result = strtok(NULL,",");
    rx = atof(result);
    result = strtok(NULL,",");
    rz = atof(result);
    result = strtok(NULL,",");
    ry = atof(result);
    printf("CMS rx,ry,rz are: %f%s%f%s%f\n", rx,",",ry,",",rz);

    result = strtok(NULL,",");
    ox = atof(result);
    result = strtok(NULL,",");
    oz = atof(result);
    result = strtok(NULL,",");
    oy = atof(result);
    printf("RgnCtrX,RgnCtrY,RgnCtrZ are: %f%s%f%s%f\n", ox,",",oy,",",oz);

    result = strtok(NULL,",");
    nx = atoi(result);
    result = strtok(NULL,",");
    nz = atoi(result);
    result = strtok(NULL,",");
    ny = atoi(result);
    printf("nx,ny,nz are: %d%s%d%s%d\n", nx,",",ny,",",nz);

    //////////////////dx,dy,and dz are the element spacings
    dx = rx/(nx-1);
    dy = ry/(ny-1);
    dz = rz/(nz-1);
    printf("dx,dy,dz are: %f%s%f%s%f\n", dx,",",dy,",",dz);

    /////////////////////////////////

    //////////////////topx,topy,and topz are the offset
    topx = ox-(rx/2);
    topy = oy-(ry/2);
    topz = -oz-(rz/2);
    printf("topx,topy,topz are: %f%s%f%s%f\n", topx,",",topy,",",topz);

    offset = ftell (ifp);
    printf ("My offset is %ld\n", offset);
    printf ("This far from the end %d\n", nx*ny*nz*4);
    fseek (ifp, -nx*ny*nz*4, SEEK_END);

    offset = ftell (ifp);
    printf ("My offset is %ld\n", offset);


    for (j = 0; j < ny; j++) {
	for (k = 0; k < nz; k++) {
	    for (i = 0; i < nx; i++) {
		float tmp_dose;
		fread (&dose, 4, 1, ifp);
		int_endian (&dose);
		tmp_dose = dose * xio_dose_scale;
		dose = (int) (tmp_dose / 10000.);
		data[k][j][i]=(short)dose;
	    }
	}
    }

    offset = ftell (ifp);
    printf ("CHECKPOINT!\n");
    printf ("My offset is %ld\n", offset);

    fclose(ifp);

    ofp = fopen(argv[2], "w");
    if(ofp==NULL) {
	printf ("Error: can't create file for writing.\n");
	return 1;
    }

    fprintf(ofp,"ObjectType = Image\n");
    fprintf(ofp,"NDims = 3\n");
    fprintf(ofp,"BinaryData = True\n");
    fprintf(ofp,"BinaryDataByteOrderMSB = False\n");
    printf("Writing topx,topy,topz...which are: %f%s%f%s%f\n", topx,",",topy,",",topz);
    fprintf(ofp,"Offset = %f %f %f\n", topx,topz,topy);   //OFFSET
    fprintf(ofp,"ElementSpacing = %f %f %f\n", dx,dz,dy); //ELEMENT SPACING
    fprintf(ofp,"DimSize = %d %d %d\n", nx,nz,ny);        //DIMENSION sIZE # of voxels

    fprintf(ofp,"AnatomicalOrientation = RAI\n");
    fprintf(ofp,"TransformMatrix = 1 0 0 0 1 0 0 0 1\n");
    fprintf(ofp,"CenterOfRotation = 0 0 0\n");
    fprintf(ofp,"ElementType = MET_SHORT\n");
    fprintf(ofp,"ElementDataFile = LOCAL\n");

    for(j = 0; j < ny; j++){
	for(k = nz-1; k >= 0; k--){ 
	    for(i = 0; i < nx; i++){ 
		fwrite(&(data[k][j][i]), 2, 1, ofp);
		//fprintf (ofp, "%d %d %d\n", k, j, i);
	    }
	}
    }
    fclose(ofp);

    return 0;
}
