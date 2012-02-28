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
    Rasterizer ();
    ~Rasterizer ();
  public:
    bool want_prefix_imgs;
    bool want_labelmap;
    bool want_ss_img;

    /* The default behavior is to "or" overlapping contours.  Set 
       this member to true if you want to "xor" instead */
    bool xor_overlapping;

    float origin[3];
    float spacing[3];
    size_t dim[3];

    unsigned char* acc_img;
    Volume* uchar_vol;
    Volume* labelmap_vol;

#if defined (commentout)
#if (PLM_USE_SS_IMAGE_VEC)
    UCharVecImageType::Pointer m_ss_img;
#else
    Volume* ss_img_vol;
#endif
#endif
    Plm_image* m_ss_img;
    bool m_use_ss_img_vec;

    size_t curr_struct_no;
    int curr_bit;

  public:
    void rasterize (
	Rtss_polyline_set *cxt,
	Plm_image_header *pih,
	bool want_prefix_imgs,
	bool want_labelmap,
	bool want_ss_img,
	bool use_ss_img_vec, 
	bool xor_overlapping
    );
  private:
    void init (
	Rtss_polyline_set *cxt,            /* Input */
	Plm_image_header *pih,             /* Input */
	bool want_prefix_imgs,             /* Input */
	bool want_labelmap,                /* Input */
	bool want_ss_img,                  /* Input */
        bool use_ss_img_vec,               /* Input */
	bool xor_overlapping               /* Input */
    );
    bool process_next (
	Rtss_polyline_set *cxt                          /* In/out */
    );
    const char* current_name (
	Rtss_polyline_set *cxt
    );
};

#endif
