/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_landmarks_h_
#define _bspline_landmarks_h_

#include "plm_config.h"
#include "plmbase.h"

class Bspline_parms;
class Bspline_state;

class gpuit_EXPORT Bspline_landmarks {
public:
    int num_landmarks;
    const Labeled_pointset *fixed_landmarks;
    const Labeled_pointset *moving_landmarks;
    float landmark_stiffness;
    char landmark_implementation;

    int *fixed_landmarks_p;
    int *fixed_landmarks_q;
public:
    Bspline_landmarks () {
	num_landmarks = 0;
	fixed_landmarks = 0;
	moving_landmarks = 0;
	landmark_stiffness = 1.0;
	landmark_implementation = 'a';
	fixed_landmarks_p = 0;
	fixed_landmarks_q = 0;
    }
    ~Bspline_landmarks () {
	/* Do not delete fixed_landmarks and moving_landmarks, they are 
	   owned by the caller. */
	if (fixed_landmarks_p) {
	    delete[] fixed_landmarks_p;
	}
	if (fixed_landmarks_q) {
	    delete[] fixed_landmarks_q;
	}
    }
    void set_landmarks (
	const Labeled_pointset *fixed_landmarks, 
	const Labeled_pointset *moving_landmarks);
    void initialize (const Bspline_xform* bxf);
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Bspline_landmarks*
bspline_landmarks_load (char *fixed_fn, char *moving_fn);

gpuit_EXPORT
void
bspline_landmarks_adjust (Bspline_landmarks *blm, Volume *fixed, Volume *moving);

void
bspline_landmarks_score (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform *bxf
);

gpuit_EXPORT
void bspline_landmarks_warp (
	Volume *vector_field, 
	Bspline_parms *parms,
	Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving);

gpuit_EXPORT
void bspline_landmarks_write_file (const char *fn, char *title, 
    float *coords, int n);

#if defined __cplusplus
}
#endif

#endif
