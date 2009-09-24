#ifndef _ramp_filter_h_
#define __ramp_filter_h_
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "fftw3.h"
#pragma comment (lib, "libfftw3-3.lib")
#endif 

#ifndef PI
	static const double PI = 3.14159265;
#endif

#ifndef DEGTORAD
	static const double DEGTORAD = 3.14159265 / 180.0;
#endif

#ifndef MARGIN
	static const unsigned int MARGIN = 5;
#else
	#error "MARGIN IS DEFINED"
#endif

//! In-place ramp filter for greyscale images
//! \param data The pixel data of the image
//! \param width The width of the image
//! \param height The height of the image
//! \template_param T The type of pixel in the image
//template <typename T>
void RampFilter(unsigned short * data,
				float* out,
				unsigned int width,
				unsigned int height)
{	int i,r,c;
	static unsigned int N;
	fftw_complex *in;
	fftw_complex *fft; 
	fftw_complex *ifft;
	fftw_plan fftp;
	fftw_plan ifftp;
	double* ramp;
	ramp=(double*)malloc(width*sizeof(double));
	if (ramp==NULL){
		printf("Malloca error");
		exit(1);
	}
	N= width * height;
	in    = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	fft   = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	ifft  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	
	for (r=0; r<MARGIN; ++r)
		memcpy(data+r*width,data+MARGIN*width,width*sizeof(unsigned short ));

	for (r=height-MARGIN; r<height; ++r)
		memcpy(data+r*width,data+(height-MARGIN-1)*width,width*sizeof(unsigned short ));

	for (r=0; r<height; ++r){
		for (c=0 ; c<MARGIN; ++c)
			data[r*width+c]=data[r*width+MARGIN];
		for (c=width-MARGIN ; c<width; ++c)
			data[r*width+c]=data[r*width+width-MARGIN-1];
	}  

	// Fill in
	for (i = 0; i < N; ++i)
	{
		in[i][0] = (double)(data[i]);
		in[i][0] /= 65535;
		in[i][0] = (in[i][0]==0? 1: in[i][0]);
		in[i][0] = -log(in[i][0]);
		in[i][1] = 0.0;
	}

	for (i = 0; i < (int)width / 2; ++i)
		ramp[i] = i;

	for (i = width / 2; i < (int)width; ++i)
		ramp[i] = width - i;

	for (i = 0; i < width; ++i)
		ramp[i] *= (cos(i * DEGTORAD * 360/width)+1)/2;

	for (r = 0; r < height; ++r)
	{
		fftp  = fftw_plan_dft_1d(width, in + r*width, fft + r*width, FFTW_FORWARD, FFTW_ESTIMATE);
		ifftp = fftw_plan_dft_1d(width, fft + r*width, ifft + r*width, FFTW_BACKWARD, FFTW_ESTIMATE);

		fftw_execute(fftp);

		// Apply ramp
#if 1
		for (c = 0; c < width; ++c)
		{
			fft[r*width + c][0] *= ramp[c];
			fft[r*width + c][1] *= ramp[c];
		}
#endif

		fftw_execute(ifftp);

		fftw_destroy_plan(fftp);
		fftw_destroy_plan(ifftp);
	}


	for (i = 0; i < N; ++i)
		ifft[i][0] /= width;

	for (i = 0; i < N; ++i)
		out[i] = (float)(ifft[i][0]);

	fftw_free(in);
	fftw_free(fft);
	fftw_free(ifft);
	free(ramp);
}