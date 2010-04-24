/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_macros_h_
#define _bspline_macros_h_

#include "plm_config.h"

/* These macros are used internally by the bspline algorithm */

#define CLAMP_LINEAR_INTERPOLATE_3D(mijk, mijk_f, mijk_r, li_frac_1,	\
				    li_frac_2, moving)			\
    do {								\
	clamp_linear_interpolate (mijk[0], moving->dim[0]-1,		\
				  &mijk_f[0], &mijk_r[0],		\
				  &li_frac_1[0], &li_frac_2[0]);	\
	clamp_linear_interpolate (mijk[1], moving->dim[1]-1,		\
				  &mijk_f[1], &mijk_r[1],		\
				  &li_frac_1[1], &li_frac_2[1]);	\
	clamp_linear_interpolate (mijk[2], moving->dim[2]-1,		\
				  &mijk_f[2], &mijk_r[2],		\
				  &li_frac_1[2], &li_frac_2[2]);	\
    } while (0)

#define BSPLINE_LI_VALUE(m_val, fx1, fx2, fy1, fy2, fz1, fz2, mvf, \
			 m_img, moving)				   \
    do {							   \
	float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;		   \
	float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;		   \
								   \
	m_x1y1z1 = fx1 * fy1 * fz1 * m_img[mvf];		   \
	m_x2y1z1 = fx2 * fy1 * fz1 * m_img[mvf+1];		   \
	m_x1y2z1 = fx1 * fy2 * fz1 * m_img[mvf+moving->dim[0]];		\
	m_x2y2z1 = fx2 * fy2 * fz1 * m_img[mvf+moving->dim[0]+1];	\
	m_x1y1z2 = fx1 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]]; \
	m_x2y1z2 = fx2 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1]; \
	m_x1y2z2 = fx1 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]]; \
	m_x2y2z2 = fx2 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1]; \
	m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1		\
		+ m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;		\
    } while (0)


#endif
