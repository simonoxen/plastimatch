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

int
main (int argc, char *argv[])
{
    FILE *ifp; FILE *ofp;FILE *ifp2; 

    long offset; //deleteme when done debugging (along with offset).

    char *mode = {"rc"}; /*rt vs. rc */
    char myarray[200];
    char prevline[200];
    char currentline[200];
    char *result = NULL;
    int i; int j; int k;
    double CMS_rx; double CMS_ry; double CMS_rz;	double MHA_rx; double MHA_ry; double MHA_rz;
    double CMS_ox; double CMS_oy; double CMS_oz; double MHA_ox; double MHA_oy; double MHA_oz;
    int CMS_nPtsX; int CMS_nPtsY; int CMS_nPtsZ; int MHA_nPtsX; int MHA_nPtsY; int MHA_nPtsZ;
    double MHA_dx; double MHA_dy; double MHA_dz;//element spacing
    double MHA_startX; double MHA_startY; double MHA_startZ; //offset (top left corner of first slice)

    int dose; short dose2;

    int o, p, q, l, m, n;
    int ***data;

    printf ("Size of int = %d\n", sizeof(int));

    l = 520; n = 520; m = 520;

    data = malloc (l*sizeof(int**));
    for (o=0; o<l; o++)
    {
	data[o] = malloc (m*sizeof(int*));
	for (p=0; p<m; p++)
	{
	    data[o][p] = malloc (n*sizeof(int));
	    for (q=0; q<n; q++)
	    {
		data[o][p][q] = 0;
	    }
	}
    }


    if(argc!=4){
	printf("Usage: mha2cms.c mhafile.mha newdosename dosetemplate");
	exit(0);}
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    //ifp = fopen("dose1.mha", mode);
    ifp = fopen(argv[1], mode);
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

    if (ifp == NULL) {
	fprintf(stderr, "Can't open input file in.list!\n");
	exit(1);
    }


    for(i = 0; i < 12; i++){
	fgets(myarray, sizeof(char)*500, ifp);
	//printf("LINE: %s",array);
	result = strstr(myarray,"DimSize");
	if (result !=NULL){
	    result = strtok(result," ");
	    result = strtok(NULL," "); //skip the stuff before the equal sign
	    result = strtok(NULL," ");
	    MHA_nPtsX = atoi(result);
	    result = strtok(NULL," ");
	    MHA_nPtsY = atoi(result);
	    result = strtok(NULL," ");
	    MHA_nPtsZ = atoi(result);
	    printf("MHA nPts (x,y,z) are: %d%s%d%s%d\n", MHA_nPtsX,",",MHA_nPtsY,",",MHA_nPtsZ);
	}

	result = strstr(myarray,"ElementSpacing");
	if (result !=NULL){
	    result = strtok(result," ");
	    result = strtok(NULL," "); //skip the stuff before the equal sign
	    result = strtok(NULL," ");
	    MHA_dx = atof(result);
	    result = strtok(NULL," ");
	    MHA_dy = atof(result);
	    result = strtok(NULL," ");
	    MHA_dz = atof(result);
	    printf("MHA_dimSpacings (x,y,z) are: %f%s%f%s%f\n", MHA_dx,",",MHA_dy,",",MHA_dz);
	}

	result = strstr(myarray,"Offset");
	if (result !=NULL){
	    result = strtok(result," ");
	    result = strtok(NULL," "); //skip the stuff before the equal sign
	    result = strtok(NULL," ");
	    MHA_startX = atof(result);
	    result = strtok(NULL," ");
	    MHA_startY = atof(result);
	    result = strtok(NULL," ");
	    MHA_startZ = atof(result);
	    printf("MHA_startCoords (x,y,z) is: %f%s%f%s%f\n", MHA_startX,",",MHA_startY,",",MHA_startZ);
	}
    }


    MHA_rx = MHA_dx*(MHA_nPtsX-1); CMS_rx=MHA_rx; 
    MHA_ry = MHA_dy*(MHA_nPtsY-1); CMS_rz=MHA_ry; 
    MHA_rz = MHA_dz*(MHA_nPtsZ-1); CMS_ry=MHA_rz;	
    printf("MHA_ranges (x,y,z) are: %f%s%f%s%f\n", MHA_rx,",",MHA_ry,",",MHA_rz);
    printf("CMS_ranges (x,y,z) are: %f%s%f%s%f\n", CMS_rx,",",CMS_ry,",",CMS_rz);

    MHA_ox = MHA_startX + MHA_rx/2;
    MHA_oy = MHA_startY + MHA_ry/2;
    MHA_oz = MHA_startZ + MHA_rz/2;

    CMS_nPtsX=MHA_nPtsX; CMS_nPtsY=MHA_nPtsZ; CMS_nPtsZ=MHA_nPtsY;
    CMS_ox=MHA_ox; CMS_oy=MHA_oz; CMS_oz = -MHA_oy;

    //KeyLine = '0,'+str(CMS_rx)+','+str(CMS_rz)+','+str(CMS_ry)+','+str(CMS_ox)+','+str(CMS_oz)+','+str(CMS_oy)+','+str(CMS_nxPts)+','+str(CMS_nzPts)+','+str(CMS_nyPts)
    printf("Keyline is: %s%f%s%f%s%f%s%f%s%f%s%f%s%d%s%d%s%d\n","0,",CMS_rx,",",CMS_rz,",",CMS_ry,",",CMS_ox,",",CMS_oz,",",CMS_oy,",",CMS_nPtsX,",",CMS_nPtsZ,",",CMS_nPtsY);


    offset = ftell (ifp);
    printf ("My offset is %d\n", offset);
    printf ("This far from the end %d\n", MHA_nPtsX*MHA_nPtsY*MHA_nPtsZ*2);
    //MHA_nPtsX = 262; MHA_nPtsY = 154; MHA_nPtsZ = 193;
    fseek (ifp, -MHA_nPtsX*MHA_nPtsY*MHA_nPtsZ*2, SEEK_END);
	
    for(k = 0; k < MHA_nPtsZ; k++){
	for(j = 0; j < MHA_nPtsY; j++){
	    for(i = 0; i < MHA_nPtsX; i++){ 
		fread(&dose2, 2, 1, ifp);
		dose=dose2;
		int_endian(&dose);
		data[i][k][j]=dose;  //doesn't really matter what order i,j,k in this line is in b/c it just serves as a unique identifier for each term. what matters is the order that its read/written
	    }
	}
    }


    offset = ftell (ifp);
    printf ("My offset is %d\n", offset); 


    fclose(ifp);

    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    //ofp = fopen("myZZZdose", "w");
    ofp = fopen(argv[2], "w");
    ifp2 = fopen(argv[3], "rt");
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

    if(ofp==NULL) {
	printf("Error: can't create file for writing.\n");
	return 1;
    }
    else {
	//currentline=str(0);
	//prevline="empty";
	for(i = 0; i < 50; i++){
	    fgets(myarray, sizeof(char)*500, ifp2);
	    printf("LINE: %s",myarray);
	    strcpy(currentline,myarray);
	    printf("LINE2: %s\n",currentline);
	    //currentline=myarray;}
	    if(((currentline[0]=='0')&&(currentline[1]=='\n'))&&((prevline[0]=='0')&&(prevline[1]=='\n'))){
		fprintf(ofp,currentline);
		printf("Found two zeros!\n");
		break;}
	    fprintf(ofp,currentline);
	    strcpy(prevline,currentline);}

	for(k = 0; k < MHA_nPtsZ; k++){  
	    for(j = MHA_nPtsY-1; j >= 0; j--){ //going through the CMS Z
		for(i = 0; i < MHA_nPtsX; i++){ 
		    fwrite(&(data[i][k][j]), 4, 1, ofp);
		}
	    }
	}
	fclose(ofp);
	fclose(ifp2);
    }
    return 0;
}
