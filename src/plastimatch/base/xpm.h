/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xpm_h_
#define _xpm_h_

#include "plmbase.h"

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
	
#endif
