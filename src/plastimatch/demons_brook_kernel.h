/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demon_brook_kernel_h_
#define _demon_brook_kernel_h_

void k_epdx (::brook::stream vol_st1,
		const float  size,
		const float3  dim,
		const float  spacing,
		::brook::stream der_st);
void k_epdy (::brook::stream vol_st1,
		const float  size,
		const float3  dim,
		const float  spacing,
		::brook::stream der_st);
void k_epdz (::brook::stream vol_st1,
		const float  size,
		const float3  dim,
		const float  spacing,
		::brook::stream der_st);
void k_initial_vectors4 (::brook::stream pre_vec_st);
void k_initial_vectors1 (::brook::stream pre_vec_st);
void  k_evf_gcs (::brook::stream current_displacement,
		::brook::stream x_displacement,
		::brook::stream y_displacement,
		::brook::stream z_displacement,
		::brook::stream current_nabla,
		::brook::stream nabla_x,
		::brook::stream nabla_y,
		::brook::stream nabla_z,
		::brook::stream static_image,
		::brook::stream moving_image,
		const float  homog,
		const float  accel,
		const float  denominator_eps,
		const float  f_size,
		const float  m_size,
		const float3  f_dim,
		const float3  m_dim,
		const float3  f_offset,
		const float3  m_offset,
		const float3  f_pix_spacing,
		const float3  m_pix_spacing,
		::brook::stream result);
void  k_volume_difference (::brook::stream x_displacement,
		::brook::stream y_displacement,
		::brook::stream z_displacement,
		::brook::stream static_image,
		::brook::stream moving_image,
		const float3  dim,
		const float  size,
		::brook::stream result);
void  k_ssd (::brook::stream diff,
		::brook::stream result);
void  k_convert (::brook::stream vec,
		const float  spacing,
		::brook::stream pre_vec);

#endif
