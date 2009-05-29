#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "plm_int.h"

/* function to perform endian swaps when going from Big-Endian
 * to little-endian or vice-versa
 */
void int_endian(int *arg)
{ char lenbuf[4];
  char tmpc;
  memcpy(lenbuf,(const char *)arg,4);
  tmpc=lenbuf[0]; lenbuf[0]=lenbuf[3]; lenbuf[3]=tmpc;
  tmpc=lenbuf[1]; lenbuf[1]=lenbuf[2]; lenbuf[2]=tmpc;
  memcpy((char *)arg,lenbuf,4);
}


main(int argc, char *argv[])
    {
	FILE *ifp; FILE *ofp;

	long offset; //deleteme when done debugging.

	char *mode = {"rc"}; /*rt vs. rc */
	char array[200];char myarray[500];
	char *result = NULL;
	int nn; int i; int j; int k;
	double rx; double ry; double rz; //dimension sizes
	double ox; double oy; double oz;
	int nx; int ny; int nz;
	double dx; double dy; double dz; //element spacing
	double topx; double topy; double topz; //offset (top left corner of first slice)

	int dose;

	int o,p,q,l,m,n; short ***data;

	
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


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
	if(argc==1){
		printf("Usage: cms2mha.c cmsdose outputfile.mha");
		exit(0);}

//	for(i = 0; i < argc; i++){
//		printf("arg %d: %s\n", i, argv[i]);
//		printf("HELLO %d: %s\n", i, argv[i]);
//		printf("HELLO %d: %d\n", i, strcmp(argv[i], "hello"));
//		if (0==strcmp(argv[i], "hello")){
//			printf("HI %d: %s\n", i, argv[i]);}
//	}

//////////////////////////////////////////////////////////
	ifp = fopen(argv[1], mode);
	//ifp = fopen("april-test-dose.1", mode);
//////////////////////////////////////////////////////////




//////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
	if (ifp == NULL) {
	  fprintf(stderr, "Can't open input file in.list!\n");
	  exit(1);
	}

	for(i = 0; i < 20; i++){
		fgets(array, sizeof(char)*500, ifp);
		printf("LINE: %s",array);
		for(j = 0; j < 200; j++){
			myarray[j] = array[j];
			}
		//printf("NEW LINE AGAIN!: %s",myarray);

		//Now to check the number of comma-delineated entries in each...
		nn=0;
		result = strtok(array,",");
		while(result!=NULL) {
       			result = strtok( NULL, ",");
			nn++;
			//printf("result is: \%s\n", result );
			//printf("Current line is: %s",myarray);
			//printf("N is: %d\n\n",n);
   		}

		if(nn==10){
			printf("%s\n\n","found 9 commas");
			break;}
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
	printf ("My offset is %d\n", offset);
	printf ("This far from the end %d\n", nx*ny*nz*4);
	fseek (ifp, -nx*ny*nz*4, SEEK_END);

	offset = ftell (ifp);
	printf ("My offset is %d\n", offset);


	for(j = 0; j < ny; j++){ //for(j = 0; j < ny; j++){ 
		for(k = 0; k < nz; k++){ //for(k = 0; k < nz; k++){
			for(i = 0; i < nx; i++){ //for(i = 0; i < nx; i++){
				fread(&dose, 4, 1, ifp);
				int_endian(&dose);
				data[k][j][i]=(short)dose;
			} //CMS is 0-10,000? (divide by 100?, but fovia doesn't care.)
		}
	}

	offset = ftell (ifp);
	printf("CHECKPOINT!\n");
	printf ("My offset is %d\n", offset);

	fclose(ifp);

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
	//ofp = fopen("doseY.mha", "w");
	ofp = fopen(argv[2], "w");
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
	if(ofp==NULL) {
    		printf("Error: can't create file for writing.\n");
    		return 1;
  	}
	else {

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
	}

	//return -1;
    }

