/* =======================================================================*
   Copyright (c) 2005-2006 Massachusetts General Hospital.
   All rights reserved.
 * =======================================================================*/
#ifndef S_NCC_H
#define S_NCC_H

#include "fatm.h"
#include "scorewin.h"

typedef struct s_ncc_data {
    double weight_threshold;
    double std_dev_threshold;
} S_Ncc_Data;


void
s_ncc_initialize (struct score_win_struct* sws,
		   Image* pattern,
		   Image* pat_mask,
		   Image* signal,
		   Image* sig_mask,
		   Image_Rect& pat_window,
		   Image_Rect& sig_window);

void s_ncc_run (FATM_Options* options);
void s_ncc_free (FATM_Options* options);
void s_ncc_score_point (FATM_Options* options,
			Scorewin_Struct* ss);

#endif
