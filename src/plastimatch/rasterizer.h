/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rasterizer_h_
#define _rasterizer_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "itk_image.h"
#include "plm_image_header.h"
#include "volume.h"

class plastimatch1_EXPORT Rasterizer {
  public:
    bool want_prefix_imgs;
    bool want_labelmap;
    bool want_ss_img;

    float origin[3];
    float spacing[3];
    int dim[3];

    unsigned char* acc_img;
    Volume* uchar_vol;
    Volume* labelmap_vol;
#if (PLM_USE_SS_IMAGE_VEC)
    UCharVecImageType::Pointer m_ss_img;
#else
    Volume* ss_img_vol;
#endif

    int curr_struct_no;
    int curr_bit;
  public:
    Rasterizer () {}
    ~Rasterizer ();
  public:
    void rasterize (
	Rtss_polyline_set *cxt,
	Plm_image_header *pih,
	bool want_prefix_imgs,
	bool want_labelmap,
	bool want_ss_img
    );
  private:
    void init (
	Rtss_polyline_set *cxt,            /* Input */
	Plm_image_header *pih,             /* Input */
	bool want_prefix_imgs,             /* Input */
	bool want_labelmap,                /* Input */
	bool want_ss_img                   /* Input */
    );
    bool process_next (
	Rtss_polyline_set *cxt                          /* In/out */
    );
    const char* current_name (
	Rtss_polyline_set *cxt
    );
};

#if defined __cplusplus
extern "C" {
#endif

#if defined (commentout)
plastimatch1_EXPORT
void
cxt_to_mha_init (
    Cxt_to_mha_state *ctm_state,
    Rtss_polyline_set *cxt,
    Plm_image_header *pih,
    bool want_prefix_imgs,
    bool want_labelmap,
    bool want_ss_img
);
plastimatch1_EXPORT
bool
cxt_to_mha_process_next (
    Cxt_to_mha_state *ctm_state,
    Rtss_polyline_set *cxt
);
plastimatch1_EXPORT
const char*
cxt_to_mha_current_name (
    Cxt_to_mha_state *ctm_state,
    Rtss_polyline_set *cxt
);
plastimatch1_EXPORT
Cxt_to_mha_state*
cxt_to_mha_create (
    Rtss_polyline_set *cxt,
    Plm_image_header *pih
);
plastimatch1_EXPORT
void
cxt_to_mha_free (Cxt_to_mha_state *ctm_state);
plastimatch1_EXPORT
void
cxt_to_mha_destroy (Cxt_to_mha_state *ctm_state);
#endif

#if defined __cplusplus
}
#endif

#endif
