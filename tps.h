/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _tps_h_
#define _tps_h_

/* Incremental TPS only */
typedef struct tps_node Tps_node;
struct tps_node {
    float src_pos[3];
    float tgt_pos[3];
    float alpha;
};

typedef struct tps_xform Tps_xform;
struct tps_xform {
    struct tps_node *tps_nodes;
    int num_tps_nodes;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Tps_xform*
tps_load (char* fn);
gpuit_EXPORT void
tps_transform_point (Tps_node* tps, int num_tps_nodes, float pos[3]);
gpuit_EXPORT void
tps_free (Tps_node** tps, int *num_tps_nodes);

#if defined __cplusplus
}
#endif

#endif
