//===========================================================





//===========================================================

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>

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
    int num_vertices;
    //VERTICES* vertlist;
	float* x;
    float* y;
    float* z;
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
    int num_structures;
    STRUCTURE* slist;
   /* int skin_no;
    unsigned char* skin_image;*/
};
typedef struct data_header DATA_Header;
struct data_header {
    CT_Header ct;
    STRUCTURE_List structures;
};

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


void load_ct(DATA_Header* data_header, Program_Parms* parms)
{
    //uhm...trovare modo per leggere header ct...uhm
}

void load_structures(Program_Parms* parms, STRUCTURE_List* structures){

	FILE* fp;
	//char buf[BUFLEN];
	STRUCTURE* curr_structure=(STRUCTURE*)malloc(sizeof(STRUCTURE*));
	POLYLINE* curr_contour=(POLYLINE*)malloc(sizeof(POLYLINE*));
	//VERTICES* curr_vert=(VERTICES*)malloc(sizeof(VERTICES*));
	curr_structure->num_contours=0;
	curr_contour->num_vertices=0;

	int ord=0;
	int num_pt=0;
	int num_cn=0;
	char name_str[BUFLEN];
	char inter[BUFLEN];
	int pos=0;
	char dumm;
	int flag=0;
	int res=0;
	float x=0;
	float y=0;
	float z=0;
	//int a=0;
	
	fp=fopen(parms->file_txt,"r");
	

	if (!fp) { 
		printf ("Could not open contour file\n");
		exit(-1);
	}
	
	while(feof(fp)==0) {
		if(flag==0)
		{
			fscanf(fp,"%s",name_str);
			res=strcmp("HEADER",name_str);		
			if(res==0)
			{	
				while (fscanf(fp,"%d %s",&ord,inter)==2)
				{
					structures->num_structures++;
					structures->slist=(STRUCTURE*) realloc (structures->slist, 
					structures->num_structures*sizeof(STRUCTURE));
					curr_structure=&structures->slist[structures->num_structures];
					strcpy(curr_structure->name,inter);
				}	
				fscanf(fp,"%s",name_str);
				flag=1;
			}
		}else if(flag==1){
			fscanf(fp,"%i %i %i",&ord,&num_pt,&num_cn);
			printf("ORD: %d\n NUM PT: %d\n NUM CONTORNO: %d\n",ord,num_pt,num_cn);
			curr_structure=&structures->slist[ord];
			curr_structure->num_contours=num_cn;
			curr_structure->pslist=(POLYLINE*)realloc(curr_structure->pslist,
				(curr_structure->num_contours +1)*sizeof(POLYLINE));
			curr_contour=&curr_structure->pslist[curr_structure->num_contours];
			curr_contour->num_vertices=num_pt;
			curr_contour->x=(float*)malloc(num_pt*sizeof(float));
			curr_contour->y=(float*)malloc(num_pt*sizeof(float));
			curr_contour->z=(float*)malloc(num_pt*sizeof(float));
			//pos=0;
			for(int k=0; k<=(curr_contour->num_vertices)-1; k++)
			{
				fscanf(fp,"%f%c%f%c%f%c",&x,&dumm,&y,&dumm,&z,&dumm);
				printf("nome structure: %s num vert: %d point: %f %f %f\n",curr_structure->name,curr_contour->num_vertices,x,y,z);
				curr_contour->x[k]=x;
				curr_contour->y[k]=y;
				curr_contour->z[k]=z;
				//pos++;
			}
			/*ord=0;
			num_pt=0;
			num_cn=0;
			x=0;
			y=0;
			z=0;*/
			/*printf("LAST CONTOUR HAD %d VERTICES\n",curr_contour->num_vertices);*/
			flag=1;
		}
		
	}
printf("NUM STRUCTURES: %d\n",structures->num_structures);
		printf("LAST CONTOUR HAD %d VERTICES\n",curr_contour->num_vertices);
		printf("gratulations, we made it!");
		fclose(fp);
}

int main(int argc, char* argv[])
{
	

	 if (argc<4)
		 print_usage();
	 else
	 {
		 Program_Parms* parms=(Program_Parms*)malloc(sizeof(Program_Parms*));
		 STRUCTURE_List* structures=(STRUCTURE_List*)malloc(sizeof(STRUCTURE_List*));
		 structures->num_structures=0;
		 
		 parms->file_txt=argv[1];
		 parms->file_dicom=argv[2];
		 parms->outdir=argv[3];

		  try{
			 load_structures(parms,structures);
			   
		  }
		  catch( char * str ) {
			  printf("Exception raised: " ,"%s",str);
		  }

		
		 
		 
	 }
}