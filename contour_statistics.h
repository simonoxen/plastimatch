#ifndef _contour_statistics_h
#define _contour_statistics_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "slice_extraction.h"


/* =======================================================================*
    Definitions
 * =======================================================================*/


//typedef unsigned char UCharPixelType;
//typedef itk::Image<UCharPixelType, 3>	ImgType;
//typedef itk::Image<PixelType, 2>	intImgType;

typedef UCharImageType ImgType;
typedef itk::Image<unsigned char, 2>	intImgType;

typedef struct sliceDice SLICEDICE;
struct sliceDice {
    int num_slice;
	int first_slice;
    float* dice_list;
};

typedef struct vertices VERTICES_LIST;
struct vertices {
    int num_vertices;
	float* x;
	float* y;
	float* z;
};

typedef struct triangle TRIANGLE_LIST;
struct triangle {
    int num_triangles;
	int* first;
	int* second;
	int* third;
};

typedef struct mass MASS;
struct mass {
	float* x;
	float* y;
	float* z;
	int num_triangles;
	
};

typedef struct corr CORR;
struct corr{
	int* corrpoint_index;
	int num_points;
};

typedef struct plane PLANE;
struct plane {
	int num_planes;
	float* a0;
	float* a1;
	float* a2;
	float* a3;
};

typedef struct surface SURFACE;
struct surface{
	VERTICES_LIST vertices;
	TRIANGLE_LIST triangles;
	MASS centres;
	VERTICES_LIST MDpoints;
	CORR correspondance;
	PLANE planes;
};


void do_dice_global(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output);
void do_dice_slice(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output);
void do_cp(FILE* mesh,FILE* MDpoints, SURFACE* surface, FILE* output);
void read_obj(FILE* mesh, SURFACE* surface);
void read_MDcontours(FILE* MDpoints, SURFACE* surface);
void calculate_mass(SURFACE* surface);
void cp(SURFACE* surface);
void compute_plane(SURFACE* surface);

#endif
