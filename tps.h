/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _tps_h_
#define _tps_h_

/* Incremental TPS only */
typedef struct tps_node Tps_node;
struct tps_node {
    float src[3];   /* (x,y,z) in fixed image */
    float tgt[3];   /* (x,y,z) in moving image */
    float dxyz[3];  /* tgt - src */
    float alpha;    /* RBF weight */
};

typedef struct tps_xform Tps_xform;
struct tps_xform {

    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    
    int num_tps_nodes;           /* Num control points */
    struct tps_node *tps_nodes;  /* Control point values */
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Tps_xform*
tps_xform_alloc (void);
gpuit_EXPORT Tps_xform*
tps_xform_load (char* fn);
gpuit_EXPORT void
tps_xform_save (Tps_xform *tps_xform, char *fn);
gpuit_EXPORT void
tps_xform_free (Tps_xform *tps_xform);

gpuit_EXPORT void
tps_transform_point (Tps_node* tps, int num_tps_nodes, float pos[3]);

#if defined __cplusplus
}
#endif

#endif
