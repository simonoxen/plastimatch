/* =======================================================================*
   Copyright (c) 2005-2007 Massachusetts General Hospital.
   All rights reserved.
 * =======================================================================*/
#ifndef CLIP_PAT_H
#define CLIP_PAT_H

#include "image.h"

void
weighted_clip_pat_generate (
		Image* image,
		double length,   /* in pixels */
		double width,    /* in pixels */
		double falloff,  /* in pixels */
		double angle,    /* in radians, usu [0,pi) */
		double fc,	 /* foreground color */
		double bc,	 /* background color */
		double w_falloff /* in pixels */
		);
void
clip_pat_generate (Image* image,
		   double length,   /* in pixels */
		   double width,    /* in pixels */
		   double falloff,  /* in pixels */
		   double angle,    /* in radians, usu [0,pi) */
		   double fc,	    /* foreground color */
		   double bc	    /* background color */
		   );

#endif
