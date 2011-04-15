## MHA tools
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## rev 2
## NOT TESTED ON PYTHON 3

import numpy as np
import mha

def delete_edges(img_reg, img_ref, img_out, background):
	
	# This function deletes the patient outside edges that the registration task generates.
	# img_reg = image with edges
	# img_ref = fixed image in the registration
	# img_out = image without edges
	# background = background value (HU)
	
	(reg, reg_size, reg_spacing, reg_offset, reg_data_type)=mha.read(img_reg)
	(ref, ref_size, ref_spacing, ref_offset, ref_data_type)=mha.read(img_ref)
	
	diff=reg-ref
	reg[diff==int(np.abs(background))]=background
	
	mha.write(reg, img_out, reg_spacing, reg_offset, reg_data_type)
	
