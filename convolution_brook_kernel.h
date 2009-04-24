/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _convolution_brook_kernel_h_
#define _convolution_brook_kernel_h_

void  k_average_x (::brook::stream vec_st,
		const float3  dim,
		const float  size,
		::brook::stream result);
void  k_average_y (::brook::stream vec_st,
		const float3  dim,
		const float  size,
		::brook::stream result);
void  k_average_z (::brook::stream vec_st,
		const float3  dim,
		const float  size,
		::brook::stream result);
void  k_conv_x (::brook::stream img,
		::brook::stream ker,
		const float  ker_size,
		const float3  dim,
		const float  size,
		::brook::stream result);
void  k_conv_y (::brook::stream img,
		::brook::stream ker,
		const float  ker_size,
		const float3  dim,
		const float  size,
		::brook::stream result);
void  k_conv_z (::brook::stream img,
		::brook::stream ker,
		const float  ker_size,
		const float3  dim,
		const float  size,
		::brook::stream result);

#endif
