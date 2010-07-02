/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proj_image_dir_h_
#define _proj_image_dir_h_

#include "plm_config.h"
#include "proj_image.h"

typedef struct proj_image_dir Proj_image_dir;
struct proj_image_dir {

    char *dir;             /* Dir containing images, maybe not xml file */
    int num_proj_images;
    char **proj_image_list;

    char *xml_file;
    char *img_pat;
    char *mat_pat;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Proj_image_dir*
proj_image_dir_create (char *dir);

gpuit_EXPORT
void
proj_image_dir_destroy (Proj_image_dir *pid);

gpuit_EXPORT
Proj_image* 
proj_image_dir_load_image (Proj_image_dir* pid, int index);

gpuit_EXPORT
void
proj_image_dir_select (Proj_image_dir *pid, int first, int skip, int last);

#if defined __cplusplus
}
#endif

#endif
