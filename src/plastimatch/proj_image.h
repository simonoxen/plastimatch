/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_image_h_
#define _proj_image_h_

#include "plm_config.h"
#include "fdk_opts.h"
#include "proj_matrix.h"
#include "volume.h"

class gpuit_EXPORT Proj_image
{
public:
    Proj_image (void);
    Proj_image (const char* img_filename, const char* mat_filename);
    Proj_image (const char* img_filename, const double xy_offset[2]);
    ~Proj_image (void);

public:
    int dim[2];              /* dim[0] = cols, dim[1] = rows */
    double xy_offset[2];     /* Offset of center pixel */
    Proj_matrix *pmat;       /* Geometry of panel and source */
    float* img;		     /* Pixel data */

public:
    void clear ();
    bool have_image ();
    void init ();
    void load (const char* img_filename, const char* mat_filename);
    void load_pfm (const char* img_filename, const char* mat_filename);
    void load_raw (const char* img_filename, const char* mat_filename);
    void load_hnd (const char* img_filename);
    void set_xy_offset (const double xy_offset[2]);
};

#if defined __cplusplus
extern "C" {
#endif

#if defined (commentout)
gpuit_EXPORT
Proj_image*
proj_image_create (void);
#endif

gpuit_EXPORT 
void 
proj_image_free (Proj_image* proj);

gpuit_EXPORT 
void
proj_image_destroy (Proj_image* proj);

gpuit_EXPORT Proj_image* 
proj_image_load_and_filter (
    Fdk_options * options, 
    const char* img_filename, 
    const char* mat_filename
);

gpuit_EXPORT 
Proj_image* 
proj_image_load (
    const char* img_filename,
    const char* mat_filename
);

gpuit_EXPORT 
void
proj_image_save (
    Proj_image *proj,
    const char *img_filename,
    const char *mat_filename
);

gpuit_EXPORT
void
proj_image_filter (Proj_image *proj);

gpuit_EXPORT void
proj_image_debug_header (Proj_image *proj);

gpuit_EXPORT
void
proj_image_create_pmat (Proj_image *proj);

gpuit_EXPORT
void
proj_image_create_img (Proj_image *proj, int dim[2]);

gpuit_EXPORT void
proj_image_stats (Proj_image *proj);

#if defined __cplusplus
}
#endif

#endif
