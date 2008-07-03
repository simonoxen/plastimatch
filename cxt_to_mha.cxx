//===========================================================





//===========================================================

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
#define mkdir(a,b) _mkdir(a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "render_polyline.h"
#include "getopt.h"

#define BUFLEN 2048
#define BUF (128*1024)

typedef struct program_parms Program_Parms;
struct program_parms {
    char* file_txt;
	char* outdir;
	char* pat_number;
};

typedef struct polyline POLYLINE;
struct polyline{
    int slice_no;
	//char UID_slice[65];
    int num_vertices;
    //VERTICES* vertlist;
	float* x;
    float* y;
    //float* z;
};

typedef struct structure STRUCTURE;
struct structure {
    //int imno;
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
	//char study_ID[65];
    STRUCTURE* slist;
};
void print_usage (void)
{
	exit (-1);
	printf ("Usage: cxt_to_mha \n");
	printf ("  input text file with contours\t");
	//printf ("  output directory\n");
	printf ("  patient_number (4 digits)\n");
}


void 
load_structures(Program_Parms* parms, STRUCTURE_List* structures){

    FILE* fp;
    STRUCTURE* curr_structure=(STRUCTURE*)malloc(sizeof(STRUCTURE));
    POLYLINE* curr_contour=(POLYLINE*)malloc(sizeof(POLYLINE));

    float val_x=0;
    float val_y=0;
    float val_z=0;

    int ord=0;
    int num_pt=0;
    int num_cn=0;
    int num_slice=-1;
    char name_str[BUFLEN];
    char inter[BUFLEN];
    char tag[BUFLEN];

    char dumm;
    int flag=0;
    int res=0;
    float x=0;
    float y=0;
	

    memset(curr_structure,0,sizeof(STRUCTURE));
    memset(curr_contour,0,sizeof(POLYLINE));
    curr_structure->num_contours=0;
    curr_contour->num_vertices=0;

    fp=fopen(parms->file_txt,"r");

    if (!fp) { 
	printf ("Could not open contour file\n");
	exit(-1);
    }

    printf("Loading...");
    while(feof(fp)==0) {
	if(flag==0)
	{
	    fscanf(fp,"%s",name_str);
	    res=strcmp("HEADER",name_str);		
	    if(res==0)
	    {	
		while(fscanf(fp,"%s %f %f %f",tag,&val_x,&val_y,&val_z)==4){
		    if(strcmp("OFFSET",tag)==0){
			structures->offset[0]=val_x;
			structures->offset[1]=val_y;
			structures->offset[2]=val_z;
			//printf("%s\n",tag);
		    }else if (strcmp("DIMENSION",tag)==0){
			structures->dim[0]=val_x;
			structures->dim[1]=val_y;
			structures->dim[2]=val_z;
			//printf("%s\n",tag);
		    }else if (strcmp("SPACING",tag)==0){
			structures->spacing[0]=val_x;
			structures->spacing[1]=val_y;
			structures->spacing[2]=val_z;
			//printf("%s\n",tag);
			break;
		    }else{
			fprintf(stderr,"ERROR: Your file is not formatted correctly!");
		    }
		}
		fscanf(fp,"%s",name_str);
		if (strcmp("ROI_NAMES",name_str)!=0)
		    fprintf(stderr,"ERROR: the file parsing went wrong...can't file the tag ROI_NAMES. Please check the format!");
		while (fscanf(fp,"%d %s",&ord,inter)==2)
		{
		    structures->num_structures++;
		    structures->slist=(STRUCTURE*) realloc (structures->slist, 
							    structures->num_structures*sizeof(STRUCTURE));
		    curr_structure=&structures->slist[structures->num_structures-1];
		    strcpy(curr_structure->name,inter);
		    curr_structure->num_contours=0;
		    printf("STRUCTURE: %s\n",curr_structure->name);
		}
		//fgets(name_str, BUFLEN,fp);
		fscanf(fp,"%s",name_str);
		if (strcmp("END_OF_ROI_NAMES",name_str)!=0)
		    fprintf(stderr,"ERROR: the file parsing went wrong...can't file the tag END_OF_ROI_NAMES. Please check the format!");
		flag=1;
	    }
	    else
	    {
		fprintf(stderr,"ERROR: Your file is not formatted correctly!Can't file the HEADER string");
		exit(-1);
	    }
	}else if(flag==1){
	    if(fscanf(fp,"%d %d %d %d",&ord,&num_pt,&num_cn,&num_slice)!=4){
		break;
	    }
	    curr_structure=&structures->slist[ord-1];
	    curr_structure->num_contours=num_cn;
	    curr_structure->pslist=(POLYLINE*)realloc(curr_structure->pslist,
						      (num_cn+1)*sizeof(POLYLINE));			
	    curr_contour=&curr_structure->pslist[curr_structure->num_contours];				
	    curr_contour->num_vertices=num_pt;
	    curr_contour->slice_no=num_slice;
	    if (ord == 5) {
		printf("STRUCTURE: %d NUM_PT: %d NUM_CN: %d SLICE_NO: %d\n",ord,num_pt,num_cn,curr_contour->slice_no);
	    }
	    curr_structure->num_contours++;

	    curr_contour->x=(float*)malloc(num_pt*sizeof(float));
	    curr_contour->y=(float*)malloc(num_pt*sizeof(float));
	    if(curr_contour->y==0 || curr_contour->x==0){
		fprintf(stderr,"Error allocating memory");
		exit(-1);
	    }
	    for(int k=0; k<num_pt; k++){								
		if(fscanf(fp,"%f %f %f",&x,&y,&dumm)!=3){
		    break;
		}
		curr_contour->x[k]=x;
		curr_contour->y[k]=y;
		x=0;
		y=0;
	    }
	    ord=0;
	    num_pt=0;
	    num_cn=0;
	    flag=1;
	}
		
    }
    printf("successful!\n");
    fclose(fp);
}

int main(int argc, char* argv[])
{
	
    //printf("argc= %d\n", argc);
    if (argc<3)
	print_usage();
    else
    {
	Program_Parms* parms=(Program_Parms*)malloc(sizeof(Program_Parms));
	STRUCTURE_List* structures=(STRUCTURE_List*)malloc(sizeof(STRUCTURE_List));
	STRUCTURE* curr_structure=(STRUCTURE*)malloc(sizeof(STRUCTURE));
	POLYLINE* curr_contour=(POLYLINE*)malloc(sizeof(POLYLINE));
	STRUCTURE* curr_structure_2=(STRUCTURE*)malloc(sizeof(STRUCTURE));
	POLYLINE* curr_contour_2=(POLYLINE*)malloc(sizeof(POLYLINE));
	Volume* vol;
	FILE* fp;
	unsigned char* img;
	unsigned char* acc_img ;
	int dim[2];
	float offset[2];
	float spacing[2];
	int slice_voxels=0;
		 
		 


	memset(structures,0,sizeof(STRUCTURE_List));
	printf("Allocated Structure LIST\n");
	structures->num_structures=0;

	memset(curr_structure,0,sizeof(STRUCTURE));
	printf("Allocated Structure\n");
	memset(curr_contour,0,sizeof(POLYLINE));
	printf("Allocated Polyline\n");
	curr_structure->num_contours=0;
	curr_contour->num_vertices=0;
		 
	parms->file_txt=argv[1];
	//parms->outdir=argv[2];
	parms->pat_number=argv[2];
	//strcat(fn,parms->outdir);
	//strcat(fn,"/");
		 

	load_structures(parms,structures);

		 
	/*strcat(filename,"vertix");
	  strcat(filename,"_");*/

	for (int p=0; p < structures->num_structures; p++){
	    curr_structure_2=&structures->slist[p];
	    char filename[BUFLEN]="";
	    strcat(filename,"vertix");
	    strcat(filename,"_");
	    strcat(filename,curr_structure_2->name);
	    strcat(filename,".txt");
	    fp=fopen(filename,"w");
	    printf("FILENAME: %s\n",filename);
	    if (!fp) { 
		printf ("Could not open for writing contour file: %s\n", filename);
		exit(-1);
	    }
	    /*fprintf(fp,"NaN NaN NaN \n");*/
	    for (int r = curr_structure_2->num_contours-1; r>=0 ; r--) {
		fprintf(fp,"NaN NaN NaN \n");
		curr_contour_2=&curr_structure_2->pslist[r];
		for (int q=0; q<curr_contour_2->num_vertices; q++){
		    fprintf(fp,"%f %f %d\n",curr_contour_2->x[q],curr_contour_2->y[q],
			    curr_contour_2->slice_no);
		}
	    }
	}

	dim[0]=structures->dim[0];
	dim[1]=structures->dim[1];
	offset[0]=structures->offset[0];
	offset[1]=structures->offset[1];
	spacing[0]=structures->spacing[0];
	spacing[1]=structures->spacing[1];
	slice_voxels=dim[0]*dim[1];
	acc_img = (unsigned char*)malloc(slice_voxels*sizeof(unsigned char));
	vol=volume_create(structures->dim, structures->offset, structures->spacing, PT_UCHAR, 0);
	printf("Allocated Volume\n");
	if(vol==0){
	    fprintf(stderr,"ERROR: failed in allocating the volume"); 
	}
	img=(unsigned char*)vol->img;
	//for (int j=0; j < structures->num_structures; j++){
	for (int j=4; j <= 4; j++){
	    curr_structure=&structures->slist[j];
	    char fn[BUFLEN]="";
	    strcat(fn,parms->pat_number);
	    strcat(fn,"_");
	    strcat(fn,curr_structure->name);
	    strcat(fn,".mha");
	    printf("output filename: %s\n", fn);
	    //system("PAUSE");
	    memset (img, 0, structures->dim[0]*structures->dim[1]*structures->dim[2]*sizeof(unsigned char));
	    printf("Allocated IMG, num_contours=%d\n", curr_structure->num_contours);
	    for (int i = 0; i < curr_structure->num_contours; i++) {
		curr_contour=&curr_structure->pslist[i];
		unsigned char* slice_img = &img[curr_contour->slice_no*dim[0]*dim[1]];
		//printf ("Slice# %3d\n", curr_contour->slice_no);
		memset (acc_img, 0, dim[0]*dim[1]*sizeof(unsigned char));
		render_slice_polyline (acc_img, dim, spacing, offset, 
				       curr_contour->num_vertices, curr_contour->x, curr_contour->y);
		for (int k = 0; k < slice_voxels; k++) {
		    slice_img[k] ^= acc_img[k];
		}
	    }
	    write_mha (fn, vol);
	    //break;
	}
	volume_free(vol);
    }
}
