## MHA tools
## Author: Paolo Zaffino  (p.zaffino@unicz.it)
## rev 3
## NOT TESTED ON PYTHON 3

import numpy as np
import mha

def delete_edges(img_reg, img_ref, img_out, background):
	
	# This function deletes the patient outside edges that the registration task generates.
	# img_reg = image with edges
	# img_ref = fixed image in the registration
	# img_out = image without edges
	# background = background value (HU)
	
	reg=mha.new(input_file=img_reg)
	ref=mha.new(input_file=img_ref)
	
	diff=reg.data-ref.data
	reg.data[diff==int(np.abs(background))]=background
	
	reg.write_mha(img_out)

