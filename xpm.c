#include <stdlib.h>
#include <stdio.h>
#include "xpm.h"

void xpm_create(xpm_struct* xpm, int width, int height, int cpp)
{
	// Populate the XPM struct
	xpm->width = width;
	xpm->height = height;
	xpm->num_pix = width * height;
	xpm->num_colors = 0;
	xpm->cpp = cpp;

	// Allocate memory for pixel data
	xpm->img = (char*)malloc(width*height*sizeof(char));
}


void xpm_prime_canvas(xpm_struct* xpm, char color_code)
{
	int i;
	char* img = xpm->img;

	for (i=0; i<xpm->num_pix; i++)
		img[i] = color_code;

}

void xpm_add_color(xpm_struct* xpm, char color_code, int color)
{
	// Increase memory usage as necessary
	if (!xpm->num_colors) {
		xpm->num_colors++;
		xpm->colors = (int*)malloc(sizeof(int));
		xpm->color_code = (char*)malloc(sizeof(char));
	} else {
		xpm->num_colors++;
		xpm->colors = (int*)realloc(xpm->colors, xpm->num_colors * sizeof(int));
		xpm->color_code = (char*)realloc(xpm->color_code, xpm->num_colors * sizeof(char));
	}

	// Insert the color
	xpm->colors[xpm->num_colors - 1] = color;
	xpm->color_code[xpm->num_colors - 1] = color_code;
}

// Returns 0 on success
// -- " -- 1 on failure
// 
// I don't plan on ever needing this...
// but perhaps you will, so here it is.
int xpm_remove_color(xpm_struct* xpm, char color_code)
{
	int i;
	char* code = xpm->color_code;

	// Search for the color code and remove it
	for(i=0; i<xpm->num_colors; i++)
	{
		if (code[i] == color_code) {
			
			// Decrement palette
			xpm->num_colors--;

			// Did we remove the last color?
			if (!xpm->num_colors){
				// We have removed all the colors
				free (xpm->colors);
				free (xpm->color_code);
			} else {
				xpm->colors = (int*)realloc(xpm->colors, xpm->num_colors * sizeof(int));
				xpm->color_code = (char*)realloc(xpm->color_code, xpm->num_colors * sizeof(char));
			}

		} else {
			// color code not registered
			return 1;
		}
	}
	return 0;
}

int xpm_draw (xpm_struct* xpm, xpm_brush* brush)
{
	int i, j;
	int x1,x2,y1,y2;

	// Which brush, son?
	switch (brush->type) {
		case XPM_BOX:
			// Define bounds
			x1 = brush->x_pos;
			x2 = brush->x_pos + brush->width;
			y1 = brush->y_pos;
			y2 = brush->y_pos + brush->height;

			// Bound checking
			if ( (x1 < 0) || (x2 > xpm->width) )
				return 1;

			if ( (y1 < 0) || (y2 > xpm->height) )
				return 1;

			// Draw the box
			for (j=y1; j<y2; j++)
				for (i=x1; i<x2; i++)
					xpm->img[j * xpm->width + i] = brush->color;
		break;
	}

	return 0;
}

void xpm_write (xpm_struct* xpm, char* xpm_file)
{
	FILE *fp;
	int i,j,p;

	char* img = xpm->img;

	// Write the XPM file to disk
	if ( !(fp = fopen(xpm_file, "w")) )
		fprintf(stderr, "Error: Cannot write open XPM file for writing\n");

	// Construct the XPM header
	fprintf(fp, "static char * plm_xpm[] = {\n");
	fprintf(fp, "/* width  height  colors  cpp */\n");
	fprintf(fp, "\"%i %i %i %i\",\n\n", xpm->width, xpm->height, xpm->num_colors, xpm->cpp);

	// Construct Palette
	fprintf(fp, "/* color codes */\n");
	for (i=0; i<xpm->num_colors; i++)
		fprintf(fp, "\"%c c #%.6x\",\n", xpm->color_code[i], xpm->colors[i]);

	// Write Pixel Data
	fprintf(fp, "\n/* Pixel Data */\n");

	p=0;
	for (j=0; j<xpm->height; j++) {
		fprintf(fp, "\"");

		for (i=0; i<xpm->width; i++) {
			fprintf(fp, "%c",img[p++]);
		}
		
		fprintf(fp, "\",\n");
	}

	fprintf(fp, "};");

	// Done like dinner.
	fclose(fp);
}
