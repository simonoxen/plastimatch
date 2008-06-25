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
    char* file_dicom;
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
//typedef struct data_header DATA_Header;
//struct data_header {
//    CT_Header ct;
//    STRUCTURE_List structures;
//};

void print_usage (void)
{
	//std::cerr << "Usage: " << std::endl;
	//std::cerr << argv[0] << " input text file with contours " << " input dicom slice (need the header)" << " output directory" << std::endl;
	exit (-1);
	printf ("Usage: rtog_to_mha \n");
	printf ("  input text file with contours\t");
	printf ("  input dicom slice (need the header)\t");
	printf ("  output directory\n");
}


void 
load_structures(Program_Parms* parms, STRUCTURE_List* structures){

	FILE* fp;
	//char buf[BUFLEN];
	STRUCTURE* curr_structure=(STRUCTURE*)malloc(sizeof(STRUCTURE));
	POLYLINE* curr_contour=(POLYLINE*)malloc(sizeof(POLYLINE));
	
	int ord=0;
	int num_pt=0;
	int num_cn=0;
	char name_str[BUFLEN];
	char inter[BUFLEN];
	//char study_ID[65];
	int pos=0;
	char dumm;
	char junk[25];
	//int foo=0;
	float thick=0;
	
	int flag=0;
	int res=0;
	float x=0;
	float y=0;
	float z=0;
	
	//char buf[BUF];
	//int a=0;
	
	memset(curr_structure,0,sizeof(STRUCTURE));
	memset(curr_contour,0,sizeof(POLYLINE));
	curr_structure->num_contours=0;
	curr_contour->num_vertices=0;

	fp=fopen(parms->file_txt,"r");
	//fp=fopen(parms->file_txt,"r");

	if (!fp) { 
		printf ("Could not open contour file\n");
		exit(-1);
	}

	while(feof(fp)==0) {
		if(flag==0)
		{
			//fscanf(fp,"%s %s",name_str,study_ID);
			fscanf(fp,"%s",name_str);
			res=strcmp("HEADER",name_str);		
			if(res==0)
			{	
				fscanf(fp,"%s",study_ID);
				printf("STUDY ID is: %s\n",study_ID);
				
				while (fscanf(fp,"%d %s %s",&ord,junk,inter)==3)
				{
					
					structures->num_structures++;
					structures->slist=(STRUCTURE*) realloc (structures->slist, 
					structures->num_structures*sizeof(STRUCTURE));
					strcpy(structures->study_ID,study_ID);
					curr_structure=&structures->slist[structures->num_structures-1];
					strcpy(curr_structure->name,inter);
					curr_structure->num_contours=0;
					printf("STRUCTURE: %s\n",curr_structure->name);
				}
				printf("\n");
				strcpy(structures->study_ID,study_ID);
				printf("STUDY_ID: %s\n",structures->study_ID);
				fgets(name_str, BUFLEN,fp);
				flag=1;
			}
			else
			{
				fprintf(stderr,"ERROR: Your file is not formatted correctly!");
				exit(-1);
			}
		}else if(flag==1){
			if(fscanf(fp,"%d %f %d %d %s",&ord,&thick,&num_pt,&num_cn,junk)!=5)
			{
				break;
			}
			curr_structure=&structures->slist[ord-1];
			curr_structure->num_contours=num_cn;
			curr_structure->pslist=(POLYLINE*)realloc(curr_structure->pslist,
				(num_cn+1)*sizeof(POLYLINE));			
			curr_contour=&curr_structure->pslist[curr_structure->num_contours];				
			curr_contour->num_vertices=num_pt;
			
			fscanf(fp,"%s",inter);
			strcpy(curr_contour->UID_slice,inter);
			printf("UID: %s STRUCTURE: %d NUM_PT: %d JUNK: %s\n",curr_contour->UID_slice,ord,num_pt,junk);

			curr_contour->x=(float*)malloc(num_pt*sizeof(float));
			curr_contour->y=(float*)malloc(num_pt*sizeof(float));
			curr_contour->z=(float*)malloc(num_pt*sizeof(float));
			printf ("malloc finished!\n");
			if(curr_contour->y==0 || curr_contour->x==0 ||curr_contour->z==0 )
			{
				fprintf(stderr,"Error allocating memory");
				exit(-1);
			}
			for(int k=0; k<num_pt; k++)
			{				
				//sscanf(buf,"%f%c%f%c%f%c",&x,&dumm,&y,&dumm2,&z,&dumm3);
				
				fscanf(fp,"%f%c%f%c%f%c",&x,&dumm,&y,&dumm,&z,&dumm);
				//printf("num vert: %d point: %f %f %f\n",k,x,y,z);
				curr_contour->x[k]=x;
				curr_contour->y[k]=y;
				curr_contour->z[k]=z;
				x=0;
				y=0;
				z=0;
				//pos++;
			}
			ord=0;
			num_pt=0;
			num_cn=0;
			flag=1;
		}
		
	}
//printf("NUM STRUCTURES: %d\n",structures->num_structures);
//		printf("LAST CONTOUR HAD %d VERTICES\n",curr_contour->num_vertices);
//		printf("gratulations, we made it!");
		fclose(fp);
}

//void
//load_dicom_info(Program_Parms* parms, STRUCTURE_List* structures){
//}


int main(int argc, char* argv[])
{
	
	//printf("argc= %d\n", argc);
	 if (argc<4)
		 print_usage();
	 else
	 {
		 Program_Parms* parms=(Program_Parms*)malloc(sizeof(Program_Parms));
		 STRUCTURE_List* structures=(STRUCTURE_List*)malloc(sizeof(STRUCTURE_List));
		 memset(structures,0,sizeof(STRUCTURE_List));
		 structures->num_structures=0;
		 
		 parms->file_txt=argv[1];
		 parms->file_dicom=argv[2];
		 parms->outdir=argv[3];

		  try{
			 load_structures(parms,structures);/*
			 load_dicom_info(parms,structures);*/
			   
		  }
		  catch( char * str ) {
			  printf("Exception raised: " ,"%s",str);
		  }

		
		 
		 
	 }
}