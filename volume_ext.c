
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "volume.h"
#include "volume_ext.h"

Volume* volume_axial2coronal (Volume* ref)
{
    Volume* vout;
	int i,j,k;
    vout = volume_create (ref->dim, ref->offset, ref->pix_spacing, ref->pix_type, ref->direction_cosines, 0);
	vout->dim[1]=ref->dim[2];
	vout->dim[2]=ref->dim[1];
	vout->offset[1]=ref->offset[2];
	vout->offset[2]=ref->offset[1];
	vout->pix_spacing[1]=ref->pix_spacing[2];
	vout->pix_spacing[2]=ref->pix_spacing[1];
  
	for (k=0;k<ref->dim[2];k++){
		for (j=0;j<ref->dim[1];j++){
			memcpy ((float*)vout->img+volume_index (vout->dim, 0, (vout->dim[1]-1-k), j), (float*)ref->img+volume_index (ref->dim, 0, j, k), ref->dim[0]*ref->pix_size);
		}
	}

    return vout;
}

Volume* volume_axial2sagittal (Volume* ref)
{
    Volume* vout;
	int i,j,k;
    vout = volume_create (ref->dim, ref->offset, ref->pix_spacing, ref->pix_type, ref->direction_cosines, 0);
	vout->dim[0]=ref->dim[1];
	vout->dim[1]=ref->dim[2];
	vout->dim[2]=ref->dim[0];
	vout->offset[0]=ref->offset[1];
	vout->offset[1]=ref->offset[2];
	vout->offset[2]=ref->offset[0];
	vout->pix_spacing[0]=ref->pix_spacing[1];
	vout->pix_spacing[1]=ref->pix_spacing[2];
	vout->pix_spacing[2]=ref->pix_spacing[0];
  
	for (k=0;k<ref->dim[2];k++)
		for (j=0;j<ref->dim[1];j++)
				for (i=0;i<ref->dim[0];i++)
						memcpy ((float*)vout->img+volume_index (vout->dim, j, (vout->dim[1]-1-k), i), (float*)ref->img+volume_index (ref->dim, i, j, k), ref->pix_size);
		
	

    return vout;
}


