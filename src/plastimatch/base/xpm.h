/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xpm_h_
#define _xpm_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

enum xpm_brushes {
    XPM_BOX,
    XPM_CIRCLE
};

typedef struct xpm_brush_s xpm_brush;
typedef struct xpm_struct_s xpm_struct;

C_API xpm_struct* xpm_create(int width, int height, int cpp);
C_API void xpm_destroy (xpm_struct* xpm);
C_API void xpm_prime_canvas(xpm_struct* xpm, char color_code);
C_API void xpm_add_color(xpm_struct* xpm, char color_code, int color);
C_API int xpm_remove_color(xpm_struct* xpm, char color_code);
C_API int xpm_draw (xpm_struct* xpm, xpm_brush* brush);
C_API void xpm_write (xpm_struct* xpm, char* xpm_file);
C_API xpm_brush* xpm_brush_create ();
C_API void xpm_brush_destroy (xpm_brush *brush);
C_API void xpm_brush_set_type (xpm_brush* brush, xpm_brushes type);
C_API void xpm_brush_set_color (xpm_brush* brush, char color);
C_API void xpm_brush_set_pos (xpm_brush *brush, int x, int y);
C_API void xpm_brush_dec_x_pos (xpm_brush *brush, int x);
C_API void xpm_brush_dec_y_pos (xpm_brush *brush, int y);
C_API void xpm_brush_inc_x_pos (xpm_brush *brush, int x);
C_API void xpm_brush_inc_y_pos (xpm_brush *brush, int y);
C_API void xpm_brush_set_x_pos (xpm_brush *brush, int x);
C_API void xpm_brush_set_y_pos (xpm_brush *brush, int y);
C_API void xpm_brush_set_width (xpm_brush* brush, int width);
C_API void xpm_brush_set_height (xpm_brush* brush, int height);

	
#endif
