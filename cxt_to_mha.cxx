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


typedef struct vertices VERTICES;
struct vertices {
   /* int num_vertices;*/
    float* x;
    float* y;
    float* z;
};

typedef struct polyline POLYLINE;
struct polyline{
    int slice_no;
    int num_vertices;
    VERTICES* vertlist;
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
	VERTICES* curr_vert=(VERTICES*)malloc(sizeof(VERTICES*));
	
	int ord=0;
	int num_pt=0;
	int num_cn=0;
	char name_str[BUFLEN];
	int pos=0;
	char dumm;
	int a=0;
	
	fp=fopen(parms->file_txt,"r");
	

	if (!fp) { 
		printf ("Could not open contour file\n");
		exit(-1);
	}

	while (feof(fp)==0) {
		try{
		if(fscanf(fp,"%n%s",&ord,name_str)==2){
			system("PAUSE");
			structures->num_structures++;
			structures->slist=(STRUCTURE*) realloc (structures->slist, 
				structures->num_structures*sizeof(STRUCTURE));
			curr_structure=&structures->slist[structures->num_structures];
			strcpy(curr_structure->name,name_str);
		}else if(fscanf(fp,"%s",name_str)==1){
			//printf("sto per caricare i punti");
			a=a+1;
			printf("%n",a);
			system("PAUSE");
		}else if(fscanf(fp,"%n%n%n",&ord,&num_pt,&num_cn)==3){
			curr_structure=&structures->slist[ord];
			curr_structure->num_contours=num_cn;
			curr_structure->pslist=(POLYLINE*)realloc(curr_structure->pslist,
				curr_structure->num_contours*sizeof(POLYLINE));
			curr_contour=&curr_structure->pslist[curr_structure->num_contours];
			curr_contour->num_vertices=num_pt;
			curr_contour->vertlist=(VERTICES*)realloc(curr_contour->vertlist, 
				(num_pt/3)*sizeof(VERTICES));
			printf("salvo");
			printf("%n",num_pt);
			printf(" vertici");
			for(int k=1; k<(num_pt/3); k=k+3)
			{
				pos++;
				curr_vert=&curr_contour->vertlist[pos];
				fscanf(fp,"%f%c%f%c%f%c",curr_vert->x,&dumm,curr_vert->y,&dumm,curr_vert->z,&dumm);
			}
		}
		}
		  catch( char * str ) {
			  printf("Exception raised: " ,"%s",str);
		  }




	}

}

int main(int argc, char* argv[])
{
	

	 if (argc<4)
		 print_usage();
	 else
	 {
		  printf("Ho abbastanza argomenti");
		  system("PAUSE");
		  Program_Parms* parms=(Program_Parms*)malloc(sizeof(Program_Parms*));
		  STRUCTURE_List* structures=(STRUCTURE_List*)malloc(sizeof(STRUCTURE_List*));
		 
		 
		 system("PAUSE");
		 parms->file_txt=argv[1];
		 parms->file_dicom=argv[2];
		 parms->outdir=argv[3];

		 printf("%s %s %s", parms->file_txt, parms->file_dicom, parms->outdir);
		  try{

			 system("PAUSE");
			 load_structures(parms,structures);
			   
		  }
		  catch( char * str ) {
			  printf("Exception raised: " ,"%s",str);
		  }

		
		 
		 
	 }

	 


	 /*STRUCTURE_List* str;
	 str->num_structures=3;
	 str->slist[1]->num_contours=5;
	 str->slist[1]->pslist[1]->*/
    /*parse_args (&parms, argc, argv);*/
	/*load_ct();
		load_structure();
		render_structure();*/
}