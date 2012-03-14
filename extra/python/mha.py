########################################################################
## These functions read and write mha files (images or vector fields)
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## Rev 14
## NOT TESTED ON PYTHON 3
########################################################################

import numpy as np

####################### WRITE MHA - START - ############################

## This function writes a mha file
##
## INPUT PARAMETERS:
## A=3D/4D matrix
## fn=file name
## spacing=voxel size
## offset=spatial offset of raw data
## data_type='short', 'float' or 'uchar'

def write (A, fn, spacing, offset, data_type):
	
	
	if fn.endswith('.mha'): ## Check if the file extension is ".mha"
		
		## Check if the input matrix is an image or a vf
		if A.ndim == 3:
			data='img'
		elif A.ndim == 4:
			data='vf'
			
		f=open(fn, 'wb')
		
		## Write mha header
		f.write('ObjectType = Image\n')
		f.write('NDims = 3\n')
		f.write('BinaryData = True\n')
		f.write('BinaryDataByteOrderMSB = False\n')
		f.write('CompressedData = False\n')
		f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
		f.write('Offset = '+str(offset[0])+' '+str(offset[1])+' '+str(offset[2])+'\n')
		f.write('CenterOfRotation = 0 0 0\n')
		f.write('AnatomicalOrientation = RAI\n')
		f.write('ElementSpacing = '+str(spacing[0])+' '+str(spacing[1])+' '+str(spacing[2])+'\n')
		f.write('DimSize = '+str(A.shape[0])+' '+str(A.shape[1])+ ' '+str(A.shape[2])+'\n')
		if data == 'vf':
			f.write('ElementNumberOfChannels = 3\n')
			A=_shiftdim(A, 3) ## Shift dimensions if the input matrix is a vf
		if data_type == 'short':
			f.write('ElementType = MET_SHORT\n')
		elif data_type == 'float':
			f.write('ElementType = MET_FLOAT\n')
		elif data_type == 'uchar':
			f.write('ElementType = MET_UCHAR\n')
		f.write('ElementDataFile = LOCAL\n')
		
		## Write matrix
		f.write(A)
				
		f.close()
		
	else: ## File extension is not ".mha"
		print ('The input file name is not a mha file!')

######################## WRITE MHA - END - #############################

	 
######################## READ MHA - START - ############################


## This function reads a mha file
##
## INPUT PARAMETER:
## fn=file name
##
## RETURNED PARAMETERS:
## raw=3D/4D matrix
## siz=matrix size
## spacing=voxel size
## offset=spatial offset of raw data
## data_type='short', 'float' or 'uchar'

def read(fn):
	
	## Utility function - START -
	def __cast2int (l):
		for i in range(3):
			if l[i].is_integer():
				l[i]=int(l[i])
		return l
	## Utility function - END -
	
	if fn.endswith('.mha'): ## Check if the file extension is ".mha"
		
		f = open(fn,'rb')
		data='img' ## On default the matrix is considered to be an image

		## Read mha header
		for r in range(20):
			
			row=f.readline()
			
			if row.startswith('Offset ='):
				row=row.split('=')[1].strip()
				offset=__cast2int(map(float, row.split()))
			elif row.startswith('ElementSpacing ='):
				row=row.split('=')[1].strip()
				spacing=__cast2int(map(float, row.split()))
			elif row.startswith('DimSize ='):
				row=row.split('=')[1].strip()
				siz=map(int, row.split())
			elif row.startswith('ElementNumberOfChannels = 3'):
				data='vf' ## The matrix is a vf
			elif row.startswith('ElementType ='):
				data_type=row.split('=')[1].strip()
			elif row.startswith('ElementDataFile ='):
				break
		
		## Read raw data
		raw=''.join(f.readlines())
		
		f.close()
		
		## Raw data from string to array
		if data_type == 'MET_SHORT':
			raw=np.fromstring(raw, dtype=np.int16)
			data_type = 'short'
		elif data_type == 'MET_FLOAT':
			raw=np.fromstring(raw, dtype=np.float32)
			data_type = 'float'
		elif data_type == 'MET_UCHAR':
			raw=np.fromstring(raw, dtype=np.uint8)
			data_type = 'uchar'
		
		## Reshape array
		if data == 'img':
			raw=raw.reshape(siz[2],siz[1],siz[0]).T
		elif data == 'vf':
			raw=raw.reshape(siz[2],siz[1],siz[0],3)
			raw=__shiftdim(raw, 3).T
			siz=siz+[3]
		
		return (raw, siz, spacing, offset, data_type)
		
	else: ## Extension file is not ".mha". It returns all null values
		print ('The input file is not a mha file!')
		raw=None
		siz=None
		spacing=None
		offset=None
		data_type=None
		return (raw, siz, spacing, offset, data_type)

######################### READ MHA - END - #############################


#################### UTILITY FUNCTION - START - ########################
############ PRIVATE UTILITY FUNCTION, NOT FOR PUBLIC USE ##############

def __shiftdim (x, n):
		return x.transpose(np.roll(range(x.ndim), -n))

##################### UTILITY FUNCTION - END - #########################
