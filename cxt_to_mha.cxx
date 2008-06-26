//===========================================================





//===========================================================

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"

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
};

typedef struct ct_header CT_Header;
struct ct_header {
    //int first_image;
    //int last_image;
    int x_spacing;
    int y_spacing;
	float z_spacing;
    float x_offset;
    float y_offset;
    float z_offset;
	int num_slices;

};


//typedef struct vertices VERTICES;
//struct vertices {
//   /* int num_vertices;*/
//    float x;
//    float y;
//    float z;
//};

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
	printf ("  output directory\n");
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
					//printf("STRUCTURE: %s\n",curr_structure->name);
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
			//printf("STRUCTURE: %d NUM_PT: %d SLICE_NO: %d\n",ord,num_pt,curr_contour->slice_no);

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
	printf("successful!");
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
		 memset(structures,0,sizeof(STRUCTURE_List));
		 structures->num_structures=0;
		 
		 parms->file_txt=argv[1];
		 parms->outdir=argv[2];

		  try{
			 load_structures(parms,structures);/*
			 load_dicom_info(parms,structures);*/
			   
		  }
		  catch( char * str ) {
			  printf("Exception raised: " ,"%s",str);
		  }

		
		 
		 
	 }
}