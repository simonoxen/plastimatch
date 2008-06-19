//===========================================================





//===========================================================

#include "render_polyline.h"


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

#define BUFLEN 2048

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


typedef struct polyline POLYLINE;
struct polyline {
    int num_vertices;
    float* x;
    float* y;
    float* z;
};

typedef struct polyline_slice POLYLINE_Slice;
struct polyline_slice {
    int slice_no;
    int num_polyline;
    POLYLINE* pllist;
};

typedef struct structure STRUCTURE;
struct structure {
    int imno;
    char name[BUFLEN];
	int num_contours;
    POLYLINE_Slice* pslist;
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

//typedef struct cxt_header CXT_header;
//struct cxt_header {
//
//}

load_ct()
{
}

load_structure(){
}

int main(int argc, char* argv[])
{
	load_ct();
		load_structure();
		render_structure();
}