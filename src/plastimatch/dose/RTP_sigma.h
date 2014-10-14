/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license informatRTP
   ----------------------------------------------------------------------- */
#ifndef _RTP_sigma_h_
#define _RTP_sigma_h_

#include "RTP_plan.h"

void compute_sigmas(RTP_plan* plan, float energy, float* sigma_max, std::string size, int* margins);

void compute_sigma_pt(Rpl_volume* sigma_vol, Rpl_volume* rpl_volume, Rpl_volume* ct_vol, RTP_plan* plan, float energy);
float compute_sigma_pt_homo(Rpl_volume* sigma_vol, Rpl_volume* rpl_vol, float energy);
float compute_sigma_pt_hetero(Rpl_volume* sigma_vol, Rpl_volume* rgl_vol, Rpl_volume* ct_vol, float energy);

void compute_sigma_source(Rpl_volume* sigma_vol, Rpl_volume* rpl_volume, RTP_plan* plan, float energy);

void compute_sigma_range_compensator(Rpl_volume* sigma_vol, Rpl_volume* rpl_volume, RTP_plan* plan, float energy, int* margins);

double get_rc_eff(double rc_over_range);

#endif