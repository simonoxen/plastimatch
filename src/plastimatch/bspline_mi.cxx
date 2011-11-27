/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif

#include "bspline.h"
#include "bspline_mi.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_opts.h"
#include "interpolate.h"
#include "logfile.h"
#include "math_util.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_macros.h"
#include "bspline_macros.h"
#include "xpm.h"
#include "mha_io.h"

/* Maximum # of bins for a vopt histogram */
#define VOPT_RES 1000

/* Some neat debug facilities
 *   Will probably move these somewhere more appropriate soon
 */
Volume*
volume_clip_intensity (Volume* vin, float bot, float top)
{
    size_t i, j, N;
    int* marks;
    float* imgin;
    float* imgout;
    float min;
    Volume* vout;

    vout = volume_clone (vin);
	imgin  = (float*) vin->img;
	imgout = (float*) vout->img;

    N=0;
    min = FLT_MAX;
    for (i=0; i<vin->npix; i++) {
        if ( (imgin[i] >= bot) && (imgin[i] <= top) ) {
            N++;
        }
        if (imgin[i] < min) {
            min = imgin[i];
        }
    }
    if (N == 0) { return NULL; }
    marks = (int*) malloc (N * sizeof (int));

    j=0;
    for (i=0; i<vin->npix; i++) {
        if ( (imgin[i] >= bot) && (imgin[i] <= top) ) {
            marks[j++] = i;
        }
        imgout[i] = min;
    }

    for (i=0; i<j; i++) {
        imgout[marks[i]] = imgin[marks[i]];
    }

    free (marks);

    return vout;
}

static void
dump_vol_clipped (char* fn, Volume* vin, float bot, float top)
{
    Volume* vout;

    vout = volume_clip_intensity (vin, bot, top);
    if (vout) {
        write_mha (fn, vout);
        delete vout;
    }
}


/* -----------------------------------------------------------------------
   Initialization and teardown
   ----------------------------------------------------------------------- */
static inline double
vopt_bin_error (int start, int end, double* s_lut, double* ssq_lut, double* cnt_lut)
{
    double sq_diff;
    double diff;
    double delta;
    double v, n;

    sq_diff = ssq_lut[end] - ssq_lut[start];
    diff = s_lut[end] - s_lut[start];
    delta = (double)end - (double)start + 1.0;
    n = cnt_lut[end] - cnt_lut[start];
    v = sq_diff - (diff*diff)/delta;

    /* Penalize solutions that have bins with less than one voxel */
    if (n < 1.0) {
        return DBL_MAX;
    }

    return v;
} 

static void
bspline_initialize_mi_bigbin (
    double* hist, 
    BSPLINE_MI_Hist_Parms* hparms, 
    Volume* vol
)
{
    size_t i;
    int idx_bin;
    float* img = (float*) vol->img;

    if (!img) {
        logfile_printf ("ERROR: trying to pre-scan empty image!\n");
        exit (-1);
    }

    /* build a quick histogram */
    for (i=0; i<vol->npix; i++) {
        idx_bin = floor ((img[i] - hparms->offset) / hparms->delta);
        if (hparms->type == HIST_VOPT) {
            idx_bin = hparms->key_lut[idx_bin];
        }
        hist[idx_bin]++;
    }

    /* look for biggest bin */
    for (i=0; i<hparms->bins; i++) {
        if (hist[i] > hist[hparms->big_bin]) {
            hparms->big_bin = i;
        }
    }
//    printf ("big_bin: %i\n", hparms->big_bin);
}

static void
bspline_initialize_mi_hist_eqsp (BSPLINE_MI_Hist_Parms* hparms, Volume* vol)
{
    size_t i;
    float min_vox, max_vox;
    float* img = (float*) vol->img;

    if (!img) {
        logfile_printf ("Error trying to create histogram from empty image\n");
        exit (-1);
    }

    min_vox = max_vox = img[0];
    for (i = 0; i < vol->npix; i++) {
        if (img[i] < min_vox) {
            min_vox = img[i];
        } else if (img[i] > max_vox) {
            max_vox = img[i];
        }
    }

    /* To avoid rounding issues, top and bottom bin are only half full */
    hparms->delta = (max_vox - min_vox) / (hparms->bins - 1);
    hparms->offset = min_vox - 0.5 * hparms->delta;
}

static void
bspline_mi_hist_vopt_dump_ranges (
    BSPLINE_MI_Hist_Parms* hparms,
    Volume* vol,
    const std::string& prefix
)
{
    FILE* fp;
    std::string fn;
    char buff[1024];
    size_t i, j;

    fn = prefix + "_vopt_ranges.txt";
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;

    printf ("Writing %s vopt debug files to disk...\n", prefix.c_str());

    int old_bin = hparms->key_lut[0];
    float left = hparms->offset;
    float right = left;
    j = 0;
    for (i=0; i<hparms->keys; i++) {
        if (hparms->key_lut[i] == old_bin) {
            right += hparms->delta;
        } else {
            fprintf (fp, "Bin %u [%6.2f .. %6.2f]\n", (unsigned int) j, 
		left, right);
            sprintf (buff, "%s_vopt_lvl_%03u.mha", prefix.c_str(), 
		(unsigned int) j);
            dump_vol_clipped (buff, vol, left, right);

            old_bin = hparms->key_lut[i];
            left = right;
            right += hparms->delta;
            j++;
        }
    }
    /* Pick up the last bin */
    fprintf (fp, "Bin %u [%6.2f .. %6.2f]\n", (unsigned int) j, 
	left, right);
    sprintf (buff, "%s_vopt_lvl_%03u.mha", prefix.c_str(), 
	(unsigned int) j);
    dump_vol_clipped (buff, vol, left, right);
    fclose (fp);
}

/* JAS - 2011.08.08
 * Experimental implementation of V-Optimal Histograms
 * ref: 
 *   http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.3195&rep=rep1&type=pdf
 */
static void
bspline_initialize_mi_hist_vopt (BSPLINE_MI_Hist_Parms* hparms, Volume* vol)
{
    size_t i, j, k;
    int idx_bin;
    size_t curr, next, bottom;
    float min_vox, max_vox;

    int* tracker;
    double* tmp_hist;
    double* tmp_avg;
    double* s_lut;
    double* ssq_lut;
    double* err_lut;
    double* cnt_lut;
    double candidate;
    float* img = (float*) vol->img;

    if (!img) {
        logfile_printf ("Error trying to create histogram from empty image\n");
        exit (-1);
    }

    hparms->keys = VOPT_RES;
    tmp_hist = (double*) malloc (hparms->keys * sizeof (double));
    tmp_avg  = (double*) malloc (hparms->keys * sizeof (double));
    memset (tmp_hist, 0, hparms->keys * sizeof (double));
    memset (tmp_avg,  0, hparms->keys * sizeof (double));

    s_lut   = (double*) malloc (hparms->keys * sizeof (double));
    ssq_lut = (double*) malloc (hparms->keys * sizeof (double));
    err_lut = (double*) malloc (hparms->bins * hparms->keys * sizeof (double));
    cnt_lut = (double*) malloc (hparms->keys* sizeof (double));
    memset (err_lut, 0, hparms->bins * hparms->keys * sizeof (double));

    tracker = (int*) malloc (hparms->bins * hparms->keys * sizeof (int));
    memset (tracker, 0, hparms->bins * hparms->keys * sizeof (int));

    /* Determine input image value range */
    min_vox = max_vox = img[0];
    for (i=1; i < vol->npix; i++) {
        if (img[i] < min_vox) {
            min_vox = img[i];
        } else if (img[i] > max_vox) {
            max_vox = img[i];
        }
    }

    /* To avoid rounding issues, top and bottom bin are only half full */
    hparms->delta = (max_vox - min_vox) / (hparms->keys - 1);
    hparms->offset = min_vox - 0.5 * hparms->delta;

    /* Construct high resolution histogram w/ bin contribution averages*/
    for (i=0; i<vol->npix; i++) {
        idx_bin = floor ((img[i] - hparms->offset) / hparms->delta);
        tmp_hist[idx_bin]++;
        tmp_avg[idx_bin] += img[i] - hparms->offset;
    }

    /* Sorted estimation table */
    for (i=0; i<hparms->keys; i++) {
        if (tmp_hist[i] > 0) {
            tmp_avg[i] = tmp_avg[i] / tmp_hist[i];
        }
        else if (i > 0) {
            tmp_avg[i] = tmp_avg[i-1];
        }
    }

    /* Create lookup tables for error computations */
    s_lut[0] = tmp_avg[0];
    ssq_lut[0] = (tmp_avg[0] * tmp_avg[0]);
    cnt_lut[0] = tmp_hist[0];
    for (i=1; i<hparms->keys; i++) {
        s_lut[i] = s_lut[i-1] + tmp_avg[i];
        ssq_lut[i] = ssq_lut[i-1] + (tmp_avg[i] * tmp_avg[i]);
        cnt_lut[i] = cnt_lut[i-1] + tmp_hist[i];
//        printf ("[%i] %f\n", i, tmp_avg[i]);
    }

    free (tmp_avg);
    free (tmp_hist);

    /* Compute the one-bin scores */
    for (i=0; i<hparms->keys; i++) {
        err_lut[i] = vopt_bin_error (0, i, s_lut, ssq_lut, cnt_lut);
    }

    /* Compute best multi-bin scores */
    for (j=1; j<hparms->bins; j++) {
        for (i=0; i<hparms->keys; i++) {

            err_lut[hparms->keys*j+i] = DBL_MAX;
            tracker[hparms->keys*j+i] = 0;

            for (k=0; k<i; k++) {
                candidate = err_lut[hparms->keys*(j-1)+k] + vopt_bin_error (k+1, i, s_lut, ssq_lut, cnt_lut);
                if (candidate <= err_lut[hparms->keys*j+i]) {
                    err_lut[hparms->keys*j+i] = candidate;
                    tracker[hparms->keys*j+i] = k;
                }
            }
        }
    }

    free (s_lut);
    free (ssq_lut);
    free (err_lut);
    free (cnt_lut);

    /* Build the linear key table */
    hparms->key_lut = (int*) malloc (hparms->keys * sizeof (int));
    memset (hparms->key_lut, 0, hparms->keys * sizeof (int));

    curr = hparms->keys-1;
    for (j=hparms->bins-1; j>=0; j--) {
        next = tracker[hparms->keys*j+curr];
        bottom = next+1;
        if (j == 0) { bottom = 0; }

//        printf ("[%i] from %i to %i\tErr: %6.2e\n",
//                  j, bottom, curr, err_lut[hparms->keys*j+curr]);
        for (i=bottom; i<=curr; i++) {
            hparms->key_lut[i] = j;
        }

        curr = next;
    }

    free (tracker);
}


static void
bspline_initialize_mi_hist (BSPLINE_MI_Hist_Parms* hparms, Volume* vol)
{
    /* If user wants more than VOPT can offer, fallback to EQSP */
    if ((hparms->bins > VOPT_RES) && (hparms->type == HIST_VOPT)) {
        printf ("WARNING: Falling back to EQSP histograms.\n"
                "         (Reason: # bins > %i)\n", VOPT_RES);
        hparms->type = HIST_EQSP;
    }

    /* Histogram type specific init procedures */
    if (hparms->type == HIST_EQSP) {
        bspline_initialize_mi_hist_eqsp (hparms, vol);
    }
    else if (hparms->type == HIST_VOPT) {
        bspline_initialize_mi_hist_vopt (hparms, vol);
    }
    else {
        fprintf (stderr, "Error: Encountered invalid histogram type.  Terminating...\n");
        exit (0);
    }
}

void
bspline_initialize_mi (Bspline_parms* parms, Volume* fixed, Volume* moving)
{
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    mi_hist->m_hist = (double*) malloc (sizeof (double) * mi_hist->moving.bins);
    mi_hist->f_hist = (double*) malloc (sizeof (double) * mi_hist->fixed.bins);
    mi_hist->j_hist = (double*) malloc (sizeof (double) * mi_hist->fixed.bins * mi_hist->moving.bins);
    bspline_initialize_mi_hist (&mi_hist->moving, moving);
    bspline_initialize_mi_hist (&mi_hist->fixed, fixed);

    if (parms->debug) {
        bspline_mi_hist_vopt_dump_ranges (&mi_hist->fixed, fixed, "fixed");
        bspline_mi_hist_vopt_dump_ranges (&mi_hist->moving, moving, "moving");
    }

    /* Initialize biggest bin trackers for OpenMP MI */
    bspline_initialize_mi_bigbin (mi_hist->f_hist, &mi_hist->fixed, fixed);
    bspline_initialize_mi_bigbin (mi_hist->m_hist, &mi_hist->moving, moving);

    /* This estimate /could/ be wrong for certain image sets */
    /* Will be auto corrected after first evaluation if incorrect */
    mi_hist->joint.big_bin = mi_hist->fixed.big_bin
                           * mi_hist->moving.bins
                           + mi_hist->moving.big_bin;
}

/* -----------------------------------------------------------------------
   MI computation functions
   ----------------------------------------------------------------------- */
inline void
clamp_quadratic_interpolate_inline (
    float ma,           /* Input: (Unrounded) pixel coordinate (in vox units) */
    long dmax,          /* Input: Maximum coordinate in this dimension */
    long maqs[3],       /* Output: x, y, or z coord of 3 pixels in moving img */
    float faqs[3]       /* Output: Fraction of interpolant for 3 voxels */
)
{
    float marf = floorf (ma + 0.5); /* marf = ma, rounded, floating */
    long mari = (long) marf;        /* mari = ma, rounded, integer */

    float t = ma - marf + 0.5;
    float t2 = t * t;
    float t22 = 0.5 * t2;

    faqs[2] = t22;
    faqs[1] = - t2 + t + 0.5;
    faqs[0] = t22 - t + 0.5;

    maqs[0] = mari - 1;
    maqs[1] = mari;
    maqs[2] = mari + 1;

    if (maqs[0] < 0) {
    maqs[0] = 0;
    if (maqs[1] < 0) {
        maqs[1] = 0;
        if (maqs[2] < 0) {
        maqs[2] = 0;
        }
    }
    } else if (maqs[2] >= dmax) {
    maqs[2] = dmax - 1;
    if (maqs[1] >= dmax) {
        maqs[1] = dmax - 1;
        if (maqs[0] >= dmax) {
        maqs[0] = dmax - 1;
        }
    }
    }
}

inline void
clamp_quadratic_interpolate_grad_inline (
    float ma,          /* Input: (Unrounded) pixel coordinate (in vox units) */
    long dmax,         /* Input: Maximum coordinate in this dimension */
    long maqs[3],      /* Output: x, y, or z coord of 3 pixels in moving img */
    float faqs[3]      /* Output: Gradient interpolant for 3 voxels */
)
{
    float marf = floorf (ma + 0.5); /* marf = ma, rounded, floating */
    long mari = (long) marf;        /* mari = ma, rounded, integer */

    float t = ma - marf + 0.5;

    faqs[0] = -1.0f + t;
    faqs[1] = -2.0f * t + 1.0f;
    faqs[2] = t;

    maqs[0] = mari - 1;
    maqs[1] = mari;
    maqs[2] = mari + 1;

    if (maqs[0] < 0) {
    faqs[0] = faqs[1] = faqs[2] = 0.0f; /* No gradient at image boundary */
    maqs[0] = 0;
    if (maqs[1] < 0) {
        maqs[1] = 0;
        if (maqs[2] < 0) {
        maqs[2] = 0;
        }
    }
    } else if (maqs[2] >= dmax) {
    faqs[0] = faqs[1] = faqs[2] = 0.0f; /* No gradient at image boundary */
    maqs[2] = dmax - 1;
    if (maqs[1] >= dmax) {
        maqs[1] = dmax - 1;
        if (maqs[0] >= dmax) {
        maqs[0] = dmax - 1;
        }
    }
    }
}

/* This function will split the amout to add between two bins (linear interp) 
    based on m_val, but one bin based on f_val. */
inline void
bspline_mi_hist_lookup (
    long j_idxs[2],     /* Output: Joint histogram indices */
    long m_idxs[2],     /* Output: Moving marginal indices */
    long f_idxs[1],     /* Output: Fixed marginal indices */
    float fxs[2],       /* Output: Fraction contribution at indices */
    BSPLINE_MI_Hist* mi_hist,   /* Input:  The histogram */
    float f_val,        /* Input:  Intensity of fixed image */
    float m_val             /* Input:  Intensity of moving image */
)
{
    size_t fl;
    float midx, midx_trunc;
    size_t ml_1, ml_2;      /* 1-d index of bin 1, bin 2 */
    float mf_1, mf_2;       /* fraction to bin 1, bin 2 */
    long f_idx; /* Index into 2-d histogram */

    /* Fixed image is binned */
    fl = (long) floor ((f_val - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    f_idx = fl * mi_hist->moving.bins;

    /* This had better not happen! */
    if (fl < 0 || fl >= mi_hist->fixed.bins) {
	fprintf (stderr, "Error: fixed image binning problem.\n"
	    "Bin %ld from val %g parms [off=%g, delt=%g, (%ld bins)]\n",
	    fl, f_val, mi_hist->fixed.offset, mi_hist->fixed.delta,
	    mi_hist->fixed.bins);
	exit (-1);
    }
    
    /* Moving image binning is interpolated (linear, not b-spline) */
    midx = ((m_val - mi_hist->moving.offset) / mi_hist->moving.delta);
    midx_trunc = floorf (midx);
    ml_1 = (long) midx_trunc;
    mf_1 = midx - midx_trunc;    // Always between 0 and 1
    ml_2 = ml_1 + 1;
    mf_2 = 1.0 - mf_1;

    if (ml_1 < 0) {
	/* This had better not happen! */
	fprintf (stderr, "Error: moving image binning problem\n");
	exit (-1);
    } else if (ml_2 >= mi_hist->moving.bins) {
	/* This could happen due to rounding */
	ml_1 = mi_hist->moving.bins - 2;
	ml_2 = mi_hist->moving.bins - 1;
	mf_1 = 0.0;
	mf_2 = 1.0;
    }

    if (mf_1 < 0.0 || mf_1 > 1.0 || mf_2 < 0.0 || mf_2 > 1.0) {
	fprintf (stderr, "Error: MI interpolation problem\n");
	exit (-1);
    }

    j_idxs[0] = f_idx + ml_1;
    j_idxs[1] = f_idx + ml_2;
    fxs[0] = mf_1;
    fxs[1] = mf_2;
    f_idxs[0] = fl;
    m_idxs[0] = ml_1;
    m_idxs[1] = ml_2;
}

/* This function will split the amout to add between two bins (linear interp) 
    based on m_val, but one bin based on f_val. */
inline void
bspline_mi_hist_add (
    BSPLINE_MI_Hist* mi_hist,   /* The histogram */
    float f_val,        /* Intensity of fixed image */
    float m_val,        /* Intensity of moving image */
    float amt               /* How much to add to histogram */
)
{
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];

    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, 
    f_val, m_val);

    fxs[0] *= amt;
    fxs[1] *= amt;

    f_hist[f_idxs[0]] += amt;       /* This is inefficient */
    m_hist[m_idxs[0]] += fxs[0];
    m_hist[m_idxs[1]] += fxs[1];
    j_hist[j_idxs[0]] += fxs[0];
    j_hist[j_idxs[1]] += fxs[1];
}

/* This algorithm uses a un-normalized score. */
static float
mi_hist_score (BSPLINE_MI_Hist* mi_hist, int num_vox)
{
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;

    size_t i, j, v;
    double fnv = (double) num_vox;
    double score = 0;
    double hist_thresh = 0.001 / (mi_hist->moving.bins * mi_hist->fixed.bins);

    /* Compute cost */
    for (i = 0, v = 0; i < mi_hist->fixed.bins; i++) {
	for (j = 0; j < mi_hist->moving.bins; j++, v++) {
	    if (j_hist[v] > hist_thresh) {
		score -= j_hist[v] 
		    * logf (fnv * j_hist[v] / (m_hist[j] * f_hist[i]));
	    }
	}
    }

    score = score / fnv;
    return (float) score;
}


/* This algorithm uses a un-normalized score. */
static float
mi_hist_score_omp (BSPLINE_MI_Hist* mi_hist, int num_vox)
{
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;

    int f_bin, m_bin, j_bin;
    double fnv = (double) num_vox;
    double score = 0;
    double hist_thresh = 0.001 / (mi_hist->moving.bins * mi_hist->fixed.bins);

    /* Compute cost */
#pragma omp parallel for reduction(-:score)
    for (j_bin=0; j_bin < (mi_hist->fixed.bins * mi_hist->moving.bins); j_bin++) {
        m_bin = j_bin % mi_hist->moving.bins;
        f_bin = j_bin / mi_hist->moving.bins;
        
        if (j_hist[j_bin] > hist_thresh) {
            score -= j_hist[j_bin] * logf(fnv * j_hist[j_bin] / (m_hist[m_bin] * f_hist[f_bin]));
        }
    }

    score = score / fnv;
    return (float) score;
}

inline float
compute_dS_dP (
    double* j_hist, 
    double* f_hist, 
    double* m_hist, 
    long* j_idxs, 
    long* f_idxs, 
    long* m_idxs, 
    float num_vox_f, 
    float* fxs, 
    float score, 
    int debug
)
{
    float dS_dP_0, dS_dP_1, dS_dP;
    const float j_hist_thresh = 0.0001f;

    if (debug) {
    fprintf (stderr, "j=[%ld %ld] (%g %g), "
         "f=[%ld] (%g), "
         "m=[%ld %ld] (%g %g), "
         "fxs = (%g %g)\n",
         j_idxs[0], j_idxs[1], j_hist[j_idxs[0]], j_hist[j_idxs[1]],
         f_idxs[0], f_hist[f_idxs[0]],
         m_idxs[0], m_idxs[1], m_hist[m_idxs[0]], m_hist[m_idxs[1]],
         fxs[0], fxs[1]);
    }

    if (j_hist[j_idxs[0]] < j_hist_thresh) {
        dS_dP_0 = 0.0f;
    } else {
        dS_dP_0 = fxs[0] * (logf((num_vox_f * j_hist[j_idxs[0]]) / (f_hist[f_idxs[0]] * m_hist[m_idxs[0]])) - score);
    }
    if (j_hist[j_idxs[1]] < j_hist_thresh) {
        dS_dP_1 = 0.0f;
    } else {
        dS_dP_1 = fxs[1] * (logf((num_vox_f * j_hist[j_idxs[1]]) / (f_hist[f_idxs[0]] * m_hist[m_idxs[1]])) - score);
    }

    dS_dP = dS_dP_0 + dS_dP_1;
    if (debug) {
        fprintf (stderr, "dS_dP %g = %g %g\n", dS_dP, dS_dP_0, dS_dP_1);
    }

    return dS_dP;
}

void dump_xpm_hist (BSPLINE_MI_Hist* mi_hist, char* file_base, int iter)
{
    size_t i, j;
    int z;
    char c;

    // Graph Properties
    int graph_offset_x = 10;
    int graph_offset_y = 10;
    int graph_padding = 20;
    int graph_bar_height = 100;
    int graph_bar_width = 5;
    int graph_bar_spacing = (int)((float)graph_bar_width * (7.0/5.0));
    int graph_color_levels = 22;

    //  int fixed_bar_height;   // max bar height (pixels)
    //  int moving_bar_height;
    int joint_color;

    float fixed_scale;  // pixels per amt
    float moving_scale;
    float joint_scale;

    float moving_max_val=0; 
    float fixed_max_val=0;
    float joint_max_val=0;

    int fixed_total_width = mi_hist->fixed.bins * graph_bar_spacing;
    int moving_total_width = mi_hist->moving.bins * graph_bar_spacing;

    int graph_moving_x_pos = graph_offset_x + graph_bar_height + graph_padding;
    int graph_moving_y_pos = graph_offset_y + fixed_total_width + graph_padding + graph_bar_height;

    int graph_fixed_x_pos = graph_offset_x;
    int graph_fixed_y_pos = graph_offset_y + fixed_total_width;

    int border_padding = 5;
    int border_width = moving_total_width + 2*border_padding;
    int border_height = fixed_total_width + 2*border_padding;
    int border_x_pos = graph_moving_x_pos - border_padding;
    int border_y_pos = graph_offset_y - border_padding + (int)((float)graph_bar_width * (2.0/5.0));

    int canvas_width = 2*graph_offset_x + graph_bar_height + moving_total_width + graph_padding;
    int canvas_height = 2*graph_offset_y + graph_bar_height + fixed_total_width + graph_padding;
    
    double *m_hist = mi_hist->m_hist;
    double *f_hist = mi_hist->f_hist;
    double *j_hist = mi_hist->j_hist;
    
    // Pull out a canvas and brush!
    xpm_struct xpm;
    xpm_brush brush;

    char filename[20];

    sprintf(filename, "%s_%04i.xpm", file_base, iter);

    // ----------------------------------------------
    // Find max value for fixed
    for(i=0; i<mi_hist->fixed.bins; i++)
	if (f_hist[i] > fixed_max_val)
	    fixed_max_val = f_hist[i];
    
    // Find max value for moving
    for(i=0; i<mi_hist->moving.bins; i++)
	if (m_hist[i] > moving_max_val)
	    moving_max_val = m_hist[i];
    
    // Find max value for joint
    // (Ignoring bin 0)
    for(j=0; j<mi_hist->fixed.bins; j++) {
	for(i=0; i<mi_hist->moving.bins; i++) {
	    if ( (i > 0) && (j > 1) )
		if (j_hist[j*mi_hist->moving.bins + i] > joint_max_val)
		    joint_max_val = j_hist[j*mi_hist->moving.bins + i];
	}
    }


    // Generate scaling factors
    fixed_scale = (float)graph_bar_height / fixed_max_val;
    moving_scale = (float)graph_bar_height / moving_max_val;
    joint_scale = (float)graph_color_levels / joint_max_val;
    // ----------------------------------------------


    
    // ----------------------------------------------
    // stretch the canvas
    xpm_create (&xpm, canvas_width, canvas_height, 1);
    
    // setup the palette
    xpm_add_color (&xpm, 'a', 0xFFFFFF);    // white
    xpm_add_color (&xpm, 'b', 0x000000);    // black
    xpm_add_color (&xpm, 'z', 0xFFCC00);    // orange

    // generate a nice BLUE->RED gradient
    c = 'c';
    z = 0x0000FF;
    for (i=0; i<(graph_color_levels+1); i++)
    {
	xpm_add_color (&xpm, c, z);

	z -= 0x00000B;      // BLUE--
	z += 0x0B0000;      //  RED++

	c = (char)((int)c + 1); // LETTER++
    }

    // Prime the XPM Canvas
    xpm_prime_canvas (&xpm, 'a');
    // ----------------------------------------------
    

    printf("Drawing Histograms... ");

    
    /* Generate Moving Histogram */
    brush.type = XPM_BOX;
    brush.color = 'b';
    brush.x_pos = graph_moving_x_pos;
    brush.y_pos = graph_moving_y_pos;
    brush.width = graph_bar_width;
    brush.height = 0;

    for(i=0; i<mi_hist->moving.bins; i++)
    {
        brush.height = (int)(m_hist[i] * moving_scale);
        brush.y_pos = graph_moving_y_pos - brush.height;
        xpm_draw(&xpm, &brush);
        brush.x_pos += graph_bar_spacing;
    }

    
    /* Generate Fixed Histogram */
    brush.type = XPM_BOX;
    brush.color = 'b';
    brush.x_pos = graph_fixed_x_pos;
    brush.y_pos = graph_fixed_y_pos;
    brush.width = 0;
    brush.height = graph_bar_width;

    for(i=0; i<mi_hist->fixed.bins; i++)
    {
        brush.width = (int)(f_hist[i] * fixed_scale);
        xpm_draw(&xpm, &brush);
        brush.y_pos -= graph_bar_spacing;
    }


    /* Generate Joint Histogram */
    brush.type = XPM_BOX;
    brush.color = 'b';
    brush.x_pos = graph_moving_x_pos;
    brush.y_pos = graph_fixed_y_pos;
    brush.width = graph_bar_width;
    brush.height = graph_bar_width;

    z = 0;
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            joint_color = (int)(j_hist[z++] * joint_scale);
            if (joint_color > 0) {
                // special handling for bin 0
                if (joint_color > graph_color_levels) {
                    //  printf ("Clamp @ P(%i,%i)\n", i, j);
                    //  brush.color = (char)(graph_color_levels + 99);
                    brush.color = 'z';
                } else {
                    brush.color = (char)(joint_color + 99);
                }
            } else {
                brush.color = 'a';
            }
            xpm_draw(&xpm, &brush);     
            brush.x_pos += graph_bar_spacing;
        }
        // get ready to render new row
        brush.x_pos = graph_moving_x_pos;
        brush.y_pos -= graph_bar_spacing;
    }

    /* Generate Joint Histogram Border */
    brush.type = XPM_BOX;       // top
    brush.color = 'b';
    brush.x_pos = border_x_pos;
    brush.y_pos = border_y_pos;
    brush.width = border_width;
    brush.height = 1;
    xpm_draw(&xpm, &brush);

    brush.width = 1;        // left
    brush.height = border_height;
    xpm_draw(&xpm, &brush);

    brush.width = border_width; // bottom
    brush.height = 1;
    brush.y_pos += border_height;
    xpm_draw(&xpm, &brush);

    brush.width = 1;        // right
    brush.height = border_height;
    brush.x_pos = border_x_pos + border_width;
    brush.y_pos = border_y_pos;
    xpm_draw(&xpm, &brush);

    printf("done.\n");
    
    // Write to file
    xpm_write (&xpm, filename);
}

/* JAS 2010.11.30
 * This is an intentionally bad idea and will be removed as soon the paper I'm
 * writing sees some ink.
 *
 * Uses CRITICAL SECTIONS instead of locks to make histogram writes thread
 * safe when employing multi-core */
#if (OPENMP_FOUND)
static inline void
bspline_mi_hist_add_pvi_8_omp_crits (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float li_1[3],           /* Fraction of interpolant in lower index */
    float li_2[3])           /* Fraction of interpolant in upper index */
{
    float w[8];
    int n[8];
    size_t idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double *f_hist = mi_hist->f_hist;
    double *m_hist = mi_hist->m_hist;
    double *j_hist = mi_hist->j_hist;


    /* Compute partial volumes from trilinear interpolation weights */
    w[0] = li_1[0] * li_1[1] * li_1[2]; // Partial Volume w0
    w[1] = li_2[0] * li_1[1] * li_1[2]; // Partial Volume w1
    w[2] = li_1[0] * li_2[1] * li_1[2]; // Partial Volume w2
    w[3] = li_2[0] * li_2[1] * li_1[2]; // Partial Volume w3
    w[4] = li_1[0] * li_1[1] * li_2[2]; // Partial Volume w4
    w[5] = li_2[0] * li_1[1] * li_2[2]; // Partial Volume w5
    w[6] = li_1[0] * li_2[1] * li_2[2]; // Partial Volume w6
    w[7] = li_2[0] * li_2[1] * li_2[2]; // Partial Volume w7

    /* Note that Sum(wN) for N within [0,7] should = 1 */

    // Calculate Point Indices for 8 neighborhood
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    // Calculate fixed histogram bin and increment it
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);

    #pragma omp critical (fixed_histogram)
    {
        f_hist[idx_fbin]++;
    }

    offset_fbin = idx_fbin * mi_hist->moving.bins;

    // Add PV weights to moving & joint histograms   
    for (idx_pv=0; idx_pv<8; idx_pv++) {

        idx_mbin = floor ((m_img[n[idx_pv]] 
		- mi_hist->moving.offset) / mi_hist->moving.delta);
        idx_jbin = offset_fbin + idx_mbin;

        if (idx_mbin != mi_hist->moving.big_bin) {
            #pragma omp critical (moving_histogram)
            {
                m_hist[idx_mbin] += w[idx_pv];
            }
        }

        if (idx_jbin != mi_hist->joint.big_bin) {
            #pragma omp critical (joint_histogram)
            {
                j_hist[idx_jbin] += w[idx_pv];
            }
        }
    }
}
#endif

/* Used locks to make histogram writes
 * thread safe when employing multi-core */
#if (OPENMP_FOUND)
static inline void
bspline_mi_hist_add_pvi_8_omp (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float li_1[3],           /* Fraction of interpolant in lower index */
    float li_2[3],           /* Fraction of interpolant in upper index */
    omp_lock_t* f_locks,
    omp_lock_t* m_locks,
    omp_lock_t* j_locks)
{
    float w[8];
    int n[8];
    size_t idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double *f_hist = mi_hist->f_hist;
    double *m_hist = mi_hist->m_hist;
    double *j_hist = mi_hist->j_hist;


    /* Compute partial volumes from trilinear interpolation weights */
    w[0] = li_1[0] * li_1[1] * li_1[2]; // Partial Volume w0
    w[1] = li_2[0] * li_1[1] * li_1[2]; // Partial Volume w1
    w[2] = li_1[0] * li_2[1] * li_1[2]; // Partial Volume w2
    w[3] = li_2[0] * li_2[1] * li_1[2]; // Partial Volume w3
    w[4] = li_1[0] * li_1[1] * li_2[2]; // Partial Volume w4
    w[5] = li_2[0] * li_1[1] * li_2[2]; // Partial Volume w5
    w[6] = li_1[0] * li_2[1] * li_2[2]; // Partial Volume w6
    w[7] = li_2[0] * li_2[1] * li_2[2]; // Partial Volume w7

    /* Note that Sum(wN) for N within [0,7] should = 1 */

    // Calculate Point Indices for 8 neighborhood
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    // Calculate fixed histogram bin and increment it
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    if (mi_hist->fixed.type == HIST_VOPT) {
        idx_fbin = mi_hist->fixed.key_lut[idx_fbin];
    }

    omp_set_lock(&f_locks[idx_fbin]);
    f_hist[idx_fbin]++;
    omp_unset_lock(&f_locks[idx_fbin]);

    offset_fbin = idx_fbin * mi_hist->moving.bins;

    // Add PV weights to moving & joint histograms   
    for (idx_pv=0; idx_pv<8; idx_pv++) {

        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        if (mi_hist->moving.type == HIST_VOPT) {
            idx_mbin = mi_hist->moving.key_lut[idx_mbin];
        }
        idx_jbin = offset_fbin + idx_mbin;

        if (idx_mbin != mi_hist->moving.big_bin) {
            omp_set_lock(&m_locks[idx_mbin]);
            m_hist[idx_mbin] += w[idx_pv];
            omp_unset_lock(&m_locks[idx_mbin]);
        }

        if (idx_jbin != mi_hist->joint.big_bin) {
            omp_set_lock(&j_locks[idx_jbin]);
            j_hist[idx_jbin] += w[idx_pv];
            omp_unset_lock(&j_locks[idx_jbin]);
        }
    }
}
#endif


static inline void
bspline_mi_hist_add_pvi_8 (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float li_1[3],           /* Fraction of interpolant in lower index */
    float li_2[3])           /* Fraction of interpolant in upper index */
{
    float w[8];
    int n[8];
    int idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double *f_hist = mi_hist->f_hist;
    double *m_hist = mi_hist->m_hist;
    double *j_hist = mi_hist->j_hist;


    /* Compute partial volumes from trilinear interpolation weights */
    w[0] = li_1[0] * li_1[1] * li_1[2]; // Partial Volume w0
    w[1] = li_2[0] * li_1[1] * li_1[2]; // Partial Volume w1
    w[2] = li_1[0] * li_2[1] * li_1[2]; // Partial Volume w2
    w[3] = li_2[0] * li_2[1] * li_1[2]; // Partial Volume w3
    w[4] = li_1[0] * li_1[1] * li_2[2]; // Partial Volume w4
    w[5] = li_2[0] * li_1[1] * li_2[2]; // Partial Volume w5
    w[6] = li_1[0] * li_2[1] * li_2[2]; // Partial Volume w6
    w[7] = li_2[0] * li_2[1] * li_2[2]; // Partial Volume w7

    /* Note that Sum(wN) for N within [0,7] should = 1 */

    // Calculate Point Indices for 8 neighborhood
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    // Calculate fixed histogram bin and increment it
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    if (mi_hist->fixed.type == HIST_VOPT) {
        idx_fbin = mi_hist->fixed.key_lut[idx_fbin];
    }
    f_hist[idx_fbin]++;

    offset_fbin = idx_fbin * mi_hist->moving.bins;

    // Add PV weights to moving & joint histograms   
    for (idx_pv=0; idx_pv<8; idx_pv++) {
        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        if (mi_hist->moving.type == HIST_VOPT) {
            idx_mbin = mi_hist->moving.key_lut[idx_mbin];
        }
        idx_jbin = offset_fbin + idx_mbin;
        m_hist[idx_mbin] += w[idx_pv];
        j_hist[idx_jbin] += w[idx_pv];
    }

}

#if defined (commentout)
static inline void
bspline_mi_hist_add_pvi_6 (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float mijk[3]
)
{
    long miqs[3], mjqs[3], mkqs[3]; /* Rounded indices */
    float fxqs[3], fyqs[3], fzqs[3];    /* Fractional values */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    const float ONE_THIRD = 1.0f / 3.0f;

    /* Compute quadratic interpolation fractions */
    clamp_quadratic_interpolate_inline (mijk[0], moving->dim[0], miqs, fxqs);
    clamp_quadratic_interpolate_inline (mijk[1], moving->dim[1], mjqs, fyqs);
    clamp_quadratic_interpolate_inline (mijk[2], moving->dim[2], mkqs, fzqs);

#if 0
    printf ("[%d %d %d], [%d %d %d], [%d %d %d]\n",
    miqs[0], miqs[1], miqs[2],
    mjqs[0], mjqs[1], mjqs[2],
    mkqs[0], mkqs[1], mkqs[2]
    );
    printf ("[%f %f %f], [%f %f %f], [%f %f %f]\n",
    fxqs[0], fxqs[1], fxqs[2], 
    fyqs[0], fyqs[1], fyqs[2], 
    fzqs[0], fzqs[1], fzqs[2]
    );
#endif

    /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * (fxqs[1] + fyqs[1] + fzqs[1]));
    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[0];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fxqs[0]);
    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[2];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fxqs[2]);
    mvf = (mkqs[1] * moving->dim[1] + mjqs[0]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fyqs[0]);
    mvf = (mkqs[1] * moving->dim[1] + mjqs[2]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fyqs[2]);
    mvf = (mkqs[0] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fzqs[0]);
    mvf = (mkqs[2] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fzqs[2]);
}
#endif

/* -----------------------------------------------------------------------
   bspline_mi_pvi_8_dc_dv
   bspline_mi_pvi_6_dc_dv

   Compute pixel contribution to gradient based on histogram change

   There are 6 or 8 correspondences between fixed and moving.  
   Each of these correspondences will update 2 or 3 histogram bins
   (other options are possible, but this is my current strategy).

   First, figure out which histogram bins this correspondence influences
   by calling bspline_mi_hist_lookup().

   Next, compute dS/dP * dP/dx. 
   dP/dx is zero outside of the 8 neighborhood.  Otherwise it is +/- 1/pixel size.
   dS/dP is 1/N (ln (N P / Pf Pm) - I)
   dS/dP is weighted by the relative contribution of the 2 histogram bins.

   dS_dx and dc_dv are really the same thing.  Just different notation.

   For dc_dv:
   We do -= instead of += because our optimizer wants to minimize 
   instead of maximize.
   The right hand side is - for "left" voxels, and + for "right" voxels.

   Use a hard threshold on j_hist[j_idxs] to prevent overflow.  This test 
   should be reconsidered, because it is theoretically unsound.

   Some trivial speedups for the future:
   The pixel size component is constant, so we can post-multiply.
   1/N is constant, so post-multiply.
   ----------------------------------------------------------------------- */
static inline void
bspline_mi_pvi_8_dc_dv (
    float dc_dv[3],                /* Output */
    BSPLINE_MI_Hist* mi_hist,      /* Input */
    Bspline_state *bst,            /* Input */
    Volume *fixed,                 /* Input */
    Volume *moving,                /* Input */
    int fv,                        /* Input */
    int mvf,                       /* Input */
    float mijk[3],                 /* Input */
    float num_vox_f,               /* Input */
    float li_1[3],                 /* Input */
    float li_2[3]                  /* Input */
)
{
    float dS_dP;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
        
    Bspline_score* ssd = &bst->ssd;
    int idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    int n[8];
    float dw[24];

    dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

    /* Calculate Point Indices for 8 neighborhood */
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    /* Pre-compute differential PV slices */
    dw[3*0+0] = (  -1 ) * li_1[1] * li_1[2];    // dw0
    dw[3*0+1] = li_1[0] * (  -1 ) * li_1[2];
    dw[3*0+2] = li_1[0] * li_1[1] * (  -1 );

    dw[3*1+0] = (  +1 ) * li_1[1] * li_1[2];    // dw1
    dw[3*1+1] = li_2[0] * (  -1 ) * li_1[2];
    dw[3*1+2] = li_2[0] * li_1[1] * (  -1 );

    dw[3*2+0] = (  -1 ) * li_2[1] * li_1[2];    // dw2
    dw[3*2+1] = li_1[0] * (  +1 ) * li_1[2];
    dw[3*2+2] = li_1[0] * li_2[1] * (  -1 );

    dw[3*3+0] = (  +1 ) * li_2[1] * li_1[2];    // dw3
    dw[3*3+1] = li_2[0] * (  +1 ) * li_1[2];
    dw[3*3+2] = li_2[0] * li_2[1] * (  -1 );

    dw[3*4+0] = (  -1 ) * li_1[1] * li_2[2];    // dw4
    dw[3*4+1] = li_1[0] * (  -1 ) * li_2[2];
    dw[3*4+2] = li_1[0] * li_1[1] * (  +1 );

    dw[3*5+0] = (  +1 ) * li_1[1] * li_2[2];    // dw5
    dw[3*5+1] = li_2[0] * (  -1 ) * li_2[2];
    dw[3*5+2] = li_2[0] * li_1[1] * (  +1 );

    dw[3*6+0] = (  -1 ) * li_2[1] * li_2[2];    // dw6
    dw[3*6+1] = li_1[0] * (  +1 ) * li_2[2];
    dw[3*6+2] = li_1[0] * li_2[1] * (  +1 );

    dw[3*7+0] = (  +1 ) * li_2[1] * li_2[2];    // dw7
    dw[3*7+1] = li_2[0] * (  +1 ) * li_2[2];
    dw[3*7+2] = li_2[0] * li_2[1] * (  +1 );


    /* Fixed image voxel's histogram index */
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    if (mi_hist->fixed.type == HIST_VOPT) {
        idx_fbin = mi_hist->fixed.key_lut[idx_fbin];
    }
    offset_fbin = idx_fbin * mi_hist->moving.bins;

    /* Partial Volume Contributions */
    for (idx_pv=0; idx_pv<8; idx_pv++) {
        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        if (mi_hist->moving.type == HIST_VOPT) {
            idx_mbin = mi_hist->moving.key_lut[idx_mbin];
        }
        idx_jbin = offset_fbin + idx_mbin;
        if (j_hist[idx_jbin] > 0.0001) {
            dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - ssd->smetric;
            dc_dv[0] -= dw[3*idx_pv+0] * dS_dP;
            dc_dv[1] -= dw[3*idx_pv+1] * dS_dP;
            dc_dv[2] -= dw[3*idx_pv+2] * dS_dP;
        }
    }

    dc_dv[0] = dc_dv[0] / num_vox_f / moving->spacing[0];
    dc_dv[1] = dc_dv[1] / num_vox_f / moving->spacing[1];
    dc_dv[2] = dc_dv[2] / num_vox_f / moving->spacing[2];


#if defined (commentout)
    for (idx_pv=0; idx_pv<8; idx_pv++) {
        printf ("dw%i [ %2.5f %2.5f %2.5f ]\n", idx_pv, dw[3*idx_pv+0], dw[3*idx_pv+1], dw[3*idx_pv+2]);
    }

    printf ("S [ %2.5f %2.5f %2.5f ]\n\n\n", dc_dv[0], dc_dv[1], dc_dv[2]);
    exit(0);
#endif
}

static inline void
bspline_mi_pvi_6_dc_dv (
    float dc_dv[3],                /* Output */
    BSPLINE_MI_Hist* mi_hist,      /* Input */
    Bspline_state *bst,            /* Input */
    Volume *fixed,                 /* Input */
    Volume *moving,                /* Input */
    int fv,                        /* Input: Index into fixed  image */
    int mvf,                       /* Input: Index into moving image (unnecessary) */
    float mijk[3],                 /* Input: ijk indices in moving image (vox) */
    float num_vox_f                /* Input: Number of voxels falling into the moving image */
)
{
    long miqs[3], mjqs[3], mkqs[3]; /* Rounded indices */
    float fxqs[3], fyqs[3], fzqs[3];    /* Fractional values */
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];
    float dS_dP;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    Bspline_score* ssd = &bst->ssd;
    int debug = 0;

    dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

    /* Compute quadratic interpolation fractions */
    clamp_quadratic_interpolate_grad_inline (mijk[0], moving->dim[0], 
    miqs, fxqs);
    clamp_quadratic_interpolate_grad_inline (mijk[1], moving->dim[1], 
    mjqs, fyqs);
    clamp_quadratic_interpolate_grad_inline (mijk[2], moving->dim[2], 
    mkqs, fzqs);

    /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[0] += - fxqs[1] * dS_dP;
    dc_dv[1] += - fyqs[1] * dS_dP;
    dc_dv[2] += - fzqs[1] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[0];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[0] += - fxqs[0] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[2];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[0] += - fxqs[2] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[0]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[1] += - fyqs[0] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[2]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[1] += - fyqs[2] * dS_dP;

    mvf = (mkqs[0] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[2] += - fzqs[0] * dS_dP;

    mvf = (mkqs[2] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
    mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
    num_vox_f, fxs, ssd->smetric, debug);
    dc_dv[2] += - fzqs[2] * dS_dP;

    dc_dv[0] = dc_dv[0] / moving->spacing[0] / num_vox_f;
    dc_dv[1] = dc_dv[1] / moving->spacing[1] / num_vox_f;
    dc_dv[2] = dc_dv[2] / moving->spacing[2] / num_vox_f;
}
    

/* -----------------------------------------------------------------------
   Scoring functions
   ----------------------------------------------------------------------- */

/* JAS 2010.11.30
 * This is an intentionally bad idea and will be removed as soon the paper I'm
 * writing sees some ink.
 * 
 * B-Spline Registration using Mutual Information
 * Implementation F (not good... only for comparison to E)
 *   -- Histograms are OpenMP accelerated
 *      (using CRITICAL SECTIONS... just to show better performance with locks)
 *   -- Uses OpenMP for Cost & dc_dv computation
 *   -- Uses methods introduced in bspline_score_g_mse
 *        to compute dc_dp more rapidly.
 */
#if (OPENMP_FOUND)
void
bspline_score_f_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    int pidx;
    float num_vox_f;
    Plm_timer timer;

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    double mhis = 0.0f;      /* Moving histogram incomplete sum */
    double jhis = 0.0f;      /* Joint  histogram incomplete sum */
    size_t i, j, zz;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

#if 0
    FILE* fp = 0;
    static int it = 0;
    char debug_fn[1024];
    if (parms->debug) {
        sprintf (debug_fn, "dump_mi_%02d.txt", it++);
        fp = fopen (debug_fn, "w");
    }
#endif

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);
    ssd->num_vox = 0;

    /* PASS 1 - Accumulate histogram */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        size_t fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        size_t mijk_f[3], mvf;      /* Floor */
        size_t mijk_r[3];           /* Round */
        size_t p[3];
        size_t q[3];
        float dxyz[3];
        size_t qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Construct the linear index within tile space */
                    qidx = volume_index (bxf->vox_per_rgn, q);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = volume_index (moving->dim, mijk_f);

                    /* Add to histogram */

                    bspline_mi_hist_add_pvi_8_omp_crits (mi_hist, fixed, moving, 
                                                   fv, mvf, li_1, li_2);
                }
            }
        }   // tile
    }   // openmp

    /* Compute num_vox and find fullest fixed hist bin */
    for (i=0; i<mi_hist->fixed.bins; i++) {
        if (f_hist[i] > f_hist[mi_hist->fixed.big_bin]) {
            mi_hist->fixed.big_bin = i;
        }
        ssd->num_vox += f_hist[i];
    }

    /* Fill in the missing histogram bin */
    for(i=0; i<mi_hist->moving.bins; i++) {
        mhis += m_hist[i];
    }
    m_hist[mi_hist->moving.big_bin] = (double)ssd->num_vox - mhis;


    /* Look for the biggest moving histogram bin */
    for(i=0; i<mi_hist->moving.bins; i++) {
        if (m_hist[i] > m_hist[mi_hist->moving.big_bin]) {
            mi_hist->moving.big_bin = i;
        }
    }


    /* Fill in the missing jnt hist bin */
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            jhis += j_hist[j*mi_hist->moving.bins + i];
        }
    }
    j_hist[mi_hist->joint.big_bin] = (double)ssd->num_vox - jhis;

    
    /* Look for the biggest joint histogram bin */
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            if (j_hist[j*mi_hist->moving.bins + i] > j_hist[mi_hist->joint.big_bin]) {
                mi_hist->joint.big_bin = j*mi_hist->moving.bins + i;
            }
        }
    }

    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->smetric = mi_hist_score_omp (mi_hist, ssd->num_vox);
    num_vox_f = (float) ssd->num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        size_t fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        size_t mijk_f[3], mvf;      /* Floor */
        size_t mijk_r[3];           /* Round */
        size_t p[3];
        size_t q[3];
        float dxyz[3];
        size_t qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Construct the linear index within tile space */
                    qidx = volume_index (bxf->vox_per_rgn, q);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = volume_index (moving->dim, mijk_f);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                        fv, mvf, mijk, num_vox_f, li_1, li_2);

                    /* Update condensed tile sets */
                    bspline_update_sets (sets_x, sets_y, sets_z,
                                         qidx, dc_dv, bxf);
                }
            }
        }   // tile

        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
                           sets_x, sets_y, sets_z,
                           pidx, bxf);
    }   // openmp

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z,
                       bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

#if 0
    if (parms->debug) {
        fclose (fp);
    }
#endif

    mse_score = mse_score / ssd->num_vox;

    ssd->time_smetric = plm_timer_report (&timer);
}
#endif



/* B-Spline Registration using Mutual Information
 * Implementation E (D is still faster)
 *   -- Histograms are OpenMP accelerated
 *      (only good on i7 core? really bad otherwise it seems...)
 *   -- Uses OpenMP for Cost & dc_dv computation
 *   -- Uses methods introduced in bspline_score_g_mse
 *        to compute dc_dp more rapidly.
 */
#if (OPENMP_FOUND)
void
bspline_score_e_mi (
    Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    int pidx;
    float num_vox_f;
    Plm_timer timer;

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    double mhis = 0.0f;      /* Moving histogram incomplete sum */
    double jhis = 0.0f;      /* Joint  histogram incomplete sum */
    size_t i, j, zz;
    omp_lock_t *f_locks, *m_locks, *j_locks;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

#if 0
    FILE* fp = 0;
    char debug_fn[1024];
    static int it = 0;
    if (parms->debug) {
        sprintf (debug_fn, "dump_mi_%02d.txt", it++);
        fp = fopen (debug_fn, "w");
    }
#endif

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);
    ssd->num_vox = 0;

    /* -- OpenMP locks for histograms --------------------- */
    f_locks = (omp_lock_t*) malloc (mi_hist->fixed.bins * sizeof(omp_lock_t));
    m_locks = (omp_lock_t*) malloc (mi_hist->moving.bins * sizeof(omp_lock_t));
    j_locks = (omp_lock_t*) malloc (mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(omp_lock_t));

#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins; i++) {
        omp_init_lock(&f_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->moving.bins; i++) {
        omp_init_lock(&m_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins * mi_hist->moving.bins; i++) {
        omp_init_lock(&j_locks[i]);
    }
    /* ---------------------------------------------------- */

    /* PASS 1 - Accumulate histogram */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        size_t fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        size_t mijk_f[3], mvf;      /* Floor */
        size_t mijk_r[3];           /* Round */
        size_t p[3];
        size_t q[3];
        float dxyz[3];
        size_t qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Construct the linear index within tile space */
                    qidx = volume_index (bxf->vox_per_rgn, q);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = volume_index (moving->dim, mijk_f);

                    /* Add to histogram */

                    bspline_mi_hist_add_pvi_8_omp (mi_hist, fixed, moving, 
            fv, mvf, li_1, li_2,
            f_locks, m_locks, j_locks);
#if defined (commentout)
#endif
                }
            }
        }   // tile
    }   // openmp

    /* Compute num_vox and find fullest fixed hist bin */
    for(i=0; i<mi_hist->fixed.bins; i++) {
        if (f_hist[i] > f_hist[mi_hist->fixed.big_bin]) {
            mi_hist->fixed.big_bin = i;
        }
        ssd->num_vox += f_hist[i];
    }

    /* Fill in the missing histogram bin */
    for(i=0; i<mi_hist->moving.bins; i++) {
        mhis += m_hist[i];
    }
    m_hist[mi_hist->moving.big_bin] = (double)ssd->num_vox - mhis;


    /* Look for the biggest moving histogram bin */
//    printf ("moving.big_bin [%i -> ", mi_hist->moving.big_bin);
    for(i=0; i<mi_hist->moving.bins; i++) {
        if (m_hist[i] > m_hist[mi_hist->moving.big_bin]) {
            mi_hist->moving.big_bin = i;
        }
    }
//    printf ("%i]\n", mi_hist->moving.big_bin);


    /* Fill in the missing jnt hist bin */
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            jhis += j_hist[j*mi_hist->moving.bins + i];
        }
    }
    j_hist[mi_hist->joint.big_bin] = (double)ssd->num_vox - jhis;

    
    /* Look for the biggest joint histogram bin */
//    printf ("joint.big_bin [%i -> ", mi_hist->joint.big_bin);
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            if (j_hist[j*mi_hist->moving.bins + i] > j_hist[mi_hist->joint.big_bin]) {
                mi_hist->joint.big_bin = j*mi_hist->moving.bins + i;
            }
        }
    }
//    printf ("%i]\n", mi_hist->joint.big_bin);
    


    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->smetric = mi_hist_score_omp (mi_hist, ssd->num_vox);
    num_vox_f = (float) ssd->num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        size_t fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        size_t mijk_f[3], mvf;      /* Floor */
        size_t mijk_r[3];           /* Round */
        size_t p[3];
        size_t q[3];
        float dxyz[3];
        size_t qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Construct the linear index within tile space */
                    qidx = volume_index (bxf->vox_per_rgn, q);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = volume_index (moving->dim, mijk_f);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                        fv, mvf, mijk, num_vox_f, li_1, li_2);

                    /* Update condensed tile sets */
                    bspline_update_sets (sets_x, sets_y, sets_z,
            qidx, dc_dv, bxf);
                }
            }
        }   // tile

        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
        sets_x, sets_y, sets_z,
        pidx, bxf);
    }   // openmp

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z,
    bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);


#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins; i++) {
        omp_destroy_lock(&f_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->moving.bins; i++) {
        omp_destroy_lock(&m_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins * mi_hist->moving.bins; i++) {
        omp_destroy_lock(&j_locks[i]);
    }

#if 0
    if (parms->debug) {
        fclose (fp);
    }
#endif

    mse_score = mse_score / ssd->num_vox;
    
    ssd->time_smetric = plm_timer_report (&timer);
}
#endif


/* B-Spline Registration using Mutual Information
 * Implementation D
 *   -- Uses OpenMP for Cost & dc_dv computation
 *   -- Uses methods introduced in bspline_score_g_mse
 *        to compute dc_dp more rapidly.
 */
void
bspline_score_d_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    size_t rijk[3];
    float diff;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float num_vox_f;
    size_t pidx;
    Plm_timer timer;
    float m_val;

    size_t fijk[3];
    size_t fv;
    float mijk[3];
    float fxyz[3];
    float mxyz[3];
    size_t mijk_f[3], mvf;      /* Floor */
    size_t mijk_r[3];           /* Round */
    size_t p[3];
    size_t q[3];
    float dxyz[3];
    size_t qidx;
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    size_t zz;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);


#if 0
    FILE* fp = 0;
    char debug_fn[1024];
    static int it = 0;
    if (parms->debug) {
        sprintf (debug_fn, "dump_mi_%02d.txt", it++);
        fp = fopen (debug_fn, "w");
    }
#endif

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);
    ssd->num_vox = 0;

    /* PASS 1 - Accumulate histogram */
    LOOP_THRU_ROI_Z (rijk, fijk, bxf) {
        p[2] = REGION_INDEX_Z (rijk, bxf);
        q[2] = REGION_OFFSET_Z (rijk, bxf);
        fxyz[2] = GET_REAL_SPACE_COORD_Z (fijk, bxf);

        LOOP_THRU_ROI_Y (rijk, fijk, bxf) {
            p[1] = REGION_INDEX_Y (rijk, bxf);
            q[1] = REGION_OFFSET_Y (rijk, bxf);
            fxyz[1] = GET_REAL_SPACE_COORD_Y (fijk, bxf);

            LOOP_THRU_ROI_X (rijk, fijk, bxf) {
                int rc;
                p[0] = REGION_INDEX_X (rijk, bxf);
                q[0] = REGION_OFFSET_X (rijk, bxf);
                fxyz[0] = GET_REAL_SPACE_COORD_X (fijk, bxf);

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Find correspondence in moving image */
                rc = bspline_find_correspondence (
		    mxyz, mijk,
		    fxyz, dxyz, moving
                );

                /* If voxel is not inside moving image */
                if (!rc) continue;

                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Find linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                /* Compute moving image intensity using linear interpolation */
                // NOTE: Not used by MI PVI8
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving
                );

#if defined (commentout)
                /* LINEAR INTERPOLATION */
                bspline_mi_hist_add (mi_hist, f_img[fv], m_val, 1.0);
#endif

                /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
                bspline_mi_hist_add_pvi_8 (
                    mi_hist, fixed, moving, 
                    fv, mvf, li_1, li_2
                );

#if defined (commentout)
                /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
                bspline_mi_hist_add_pvi_6 (
                    mi_hist, fixed, moving, 
                    fv, mvf, mijk
                );
#endif

                /* Compute intensity difference */
                diff = m_val - f_img[fv];
                mse_score += diff * diff;
                ssd->num_vox ++;

            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_TRHU_ROI_Z */


    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->smetric = mi_hist_score_omp (mi_hist, ssd->num_vox);
    num_vox_f = (float) ssd->num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    LOOP_THRU_VOL_TILES (pidx, bxf) {
        int rc;
        size_t fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        size_t mijk_f[3], mvf;      /* Floor */
        size_t mijk_r[3];           /* Round */
        size_t p[3];
        size_t q[3];
        float dxyz[3];
        size_t qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));


        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        LOOP_THRU_TILE_Z (q, bxf) {
            LOOP_THRU_TILE_Y (q, bxf) {
                LOOP_THRU_TILE_X (q, bxf) {
                    
                    /* Construct coordinates into fixed image volume */
                    GET_VOL_COORDS (fijk, p, q, bxf);
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    GET_REAL_SPACE_COORDS (fxyz, fijk, bxf);

                    /* Construct the linear index within tile space */
                    qidx = volume_index (bxf->vox_per_rgn, q);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = volume_index (fixed->dim, fijk);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = volume_index (moving->dim, mijk_f);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (
                        dc_dv, mi_hist, bst,
                        fixed, moving, 
                        fv, mvf, mijk,
                        num_vox_f, li_1, li_2
                    );

                    /* Update condensed tile sets */
                    bspline_update_sets (
                        sets_x, sets_y, sets_z,
                        qidx, dc_dv, bxf
                    );

                } /* LOOP_THRU_TILE_X */
            } /* LOOP_THRU_TILE_Y */
        } /* LOOP_THRU_TILE_Z */


        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
	    sets_x, sets_y, sets_z,
	    pidx, bxf);

    } /* LOOP_THRU_VOL_TILES */


    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result will be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

#if 0
    if (parms->debug) {
        fclose (fp);
    }
#endif

    mse_score = mse_score / ssd->num_vox;
    if (parms->debug) {
        printf ("<< MSE %3.3f >>\n", mse_score);
    }

    ssd->time_smetric = plm_timer_report (&timer);
}


/* Mutual information version of implementation "C" */
void
bspline_score_c_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    size_t rijk[3];
    size_t fijk[3], fv;
    float mijk[3];
    float fxyz[3];
    float mxyz[3];
    size_t mijk_f[3], mvf;      /* Floor */
    size_t mijk_r[3];           /* Round */
    size_t p[3];
    size_t q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float dxyz[3];
    float num_vox_f;
    size_t pidx, qidx;
    Plm_timer timer;
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float m_val;

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    size_t zz;

#if 0
    FILE* fp = 0;
    char debug_fn[1024];
    static int it = 0;
    if (parms->debug) {
    sprintf (debug_fn, "dump_mi_%02d.txt", it++);
    fp = fopen (debug_fn, "w");
    }
#endif

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    ssd->num_vox = 0;

    /* PASS 1 - Accumulate histogram */
    LOOP_THRU_ROI_Z (rijk, fijk, bxf) {
        p[2] = REGION_INDEX_Z (rijk, bxf);
        q[2] = REGION_OFFSET_Z (rijk, bxf);
        fxyz[2] = GET_REAL_SPACE_COORD_Z (fijk, bxf);

        LOOP_THRU_ROI_Y (rijk, fijk, bxf) {
            p[1] = REGION_INDEX_Y (rijk, bxf);
            q[1] = REGION_OFFSET_Y (rijk, bxf);
            fxyz[1] = GET_REAL_SPACE_COORD_Y (fijk, bxf);

            LOOP_THRU_ROI_X (rijk, fijk, bxf) {
                int rc;
                p[0] = REGION_INDEX_X (rijk, bxf);
                q[0] = REGION_OFFSET_X (rijk, bxf);
                fxyz[0] = GET_REAL_SPACE_COORD_X (fijk, bxf);

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Find correspondence in moving image */
                rc = bspline_find_correspondence (
                    mxyz, mijk, fxyz, 
                    dxyz, moving
                );

                /* If voxel is not inside moving image */
                if (!rc) continue;

                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Find linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                /* Compute moving image intensity using linear interpolation */
                /* Macro is slightly faster than function */
                // NOTE: Not used by MI PVI8
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving
                );

#if defined (commentout)
                /* LINEAR INTERPOLATION */
                bspline_mi_hist_add (mi_hist, f_img[fv], m_val, 1.0);
#endif

                /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
                bspline_mi_hist_add_pvi_8 (
                    mi_hist, fixed, moving, 
                    fv, mvf, li_1, li_2
                );

#if defined (commentout)
                /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
                bspline_mi_hist_add_pvi_6 (
                    mi_hist, fixed, moving, 
                    fv, mvf, mijk
                );
#endif

                /* Compute intensity difference */
                diff = m_val - f_img[fv];
                mse_score += diff * diff;
                ssd->num_vox++;

            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_THRU_ROI_Z */

    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->smetric = mi_hist_score (mi_hist, ssd->num_vox);
    num_vox_f = (float) ssd->num_vox;

    /* PASS 2 - Compute gradient */
    LOOP_THRU_ROI_Z (rijk, fijk, bxf) {
        p[2] = REGION_INDEX_Z (rijk, bxf);
        q[2] = REGION_OFFSET_Z (rijk, bxf);
        fxyz[2] = GET_REAL_SPACE_COORD_Z (fijk, bxf);

        LOOP_THRU_ROI_Y (rijk, fijk, bxf) {
            p[1] = REGION_INDEX_Y (rijk, bxf);
            q[1] = REGION_OFFSET_Y (rijk, bxf);
            fxyz[1] = GET_REAL_SPACE_COORD_Y (fijk, bxf);

            LOOP_THRU_ROI_X (rijk, fijk, bxf) {
                int rc;
                p[0] = REGION_INDEX_X (rijk, bxf);
                q[0] = REGION_OFFSET_X (rijk, bxf);
                fxyz[0] = GET_REAL_SPACE_COORD_X (fijk, bxf);

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Find linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* Find correspondence in moving image */
                rc = bspline_find_correspondence (
                        mxyz, mijk,
                        fxyz, dxyz,
                        moving
                );

                /* If voxel is not inside moving image */
                if (!rc) continue;

                /* LINEAR INTERPOLATION - (not implemented) */

                /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Find linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                bspline_mi_pvi_8_dc_dv (
                    dc_dv, mi_hist, bst,
                    fixed, moving, 
                    fv, mvf, mijk,
                    num_vox_f, li_1, li_2
                );

#if defined (commentout)
                /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
                bspline_mi_pvi_6_dc_dv (
                    dc_dv, mi_hist, bst,
                    fixed, moving, fv, mvf, mijk,
                    num_vox_f
                );
#endif

                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);

            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_THRU_ROI_Z */

#if 0
    if (parms->debug) {
        fclose (fp);
    }
#endif

    mse_score = mse_score / ssd->num_vox;

    ssd->time_smetric = plm_timer_report (&timer);
}
