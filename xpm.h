/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xpm_h_
#define _xpm_h_

// Data Structures
enum xpm_brushes {
	XPM_BOX,
	XPM_CIRCLE
};

typedef struct xpm_struct_s xpm_struct;
struct xpm_struct_s {
	int width;		/* Image Width                 */
	int height;		/* Image Height                */
	int num_pix;		/* Width * Height              */
	int num_colors;		/* Number of Colors in Palette */
	int cpp;		/* Characters per Pixel        */
	char* color_code;	/* User Defined Color Codes    */
	int* colors;		/* Actual Color Codes          */
	char* img;		/* Pixel Data                  */
};

typedef struct xpm_brush_s xpm_brush;
struct xpm_brush_s {
	enum xpm_brushes type;	/* Type of shape */
	char color;		/* Color Code    */
	int x_pos;		/* X Postion     */
	int y_pos;		/* Y Postion     */
	int width;		/* Width         */
	int height;             /* Height        */
	int rad;		/* Radius        */
	int hparm;		/* Misc 1        */
	int lparm;		/* Misc 2        */
};
	

// Function Prototypes
#if defined __cplusplus
extern "C" {
#endif

void xpm_create(xpm_struct* xpm, int width, int height, int cpp);
void xpm_prime_canvas(xpm_struct* xpm, char color_code);
void xpm_add_color(xpm_struct* xpm, char color_code, int color);
int xpm_remove_color(xpm_struct* xpm, char color_code);
int xpm_draw (xpm_struct* xpm, xpm_brush* brush);
void xpm_write (xpm_struct* xpm, char* xpm_file);

#if defined __cplusplus
}
#endif

#endif
