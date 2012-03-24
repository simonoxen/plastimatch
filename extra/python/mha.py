########################################################################
## These functions read and write mha files (images or vector fields)
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## Rev 15
## NOT TESTED ON PYTHON 3
########################################################################

import numpy as np

####################### WRITE MHA - START - ############################

## This function writes a mha file
##
## INPUT PARAMETERS:
## stru=data_structure, it includes:
##			raw=3D/4D matrix
##			spacing=voxel size
##			offset=spatial offset of raw data
##			data_type='short', 'float' or 'uchar'
## fn=file name

def write (stru, fn):
	
	
	if fn.endswith('.mha'): ## Check if the file extension is ".mha"
		
		## Check if the input matrix is an image or a vf
		if stru['raw'].ndim == 3:
			data='img'
		elif stru['raw'].ndim == 4:
			data='vf'
			
		f=open(fn, 'wb')
		
		## Write mha header
		f.write('ObjectType = Image\n')
		f.write('NDims = 3\n')
		f.write('BinaryData = True\n')
		f.write('BinaryDataByteOrderMSB = False\n')
		f.write('CompressedData = False\n')
		f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
		f.write('Offset = '+str(stru['offset'][0])+' '+str(stru['offset'][1])+' '+str(stru['offset'][2])+'\n')
		f.write('CenterOfRotation = 0 0 0\n')
		f.write('AnatomicalOrientation = RAI\n')
		f.write('ElementSpacing = '+str(stru['spacing'][0])+' '+str(stru['spacing'][1])+' '+str(stru['spacing'][2])+'\n')
		f.write('DimSize = '+str(stru['raw'].shape[0])+' '+str(stru['raw'].shape[1])+ ' '+str(stru['raw'].shape[2])+'\n')
		if data == 'vf':
			f.write('ElementNumberOfChannels = 3\n')
			stru['raw']=_shiftdim(stru['raw'], 3) ## Shift dimensions if the input matrix is a vf
		if stru['data_type'] == 'short':
			f.write('ElementType = MET_SHORT\n')
		elif stru['data_type'] == 'float':
			f.write('ElementType = MET_FLOAT\n')
		elif stru['data_type'] == 'uchar':
			f.write('ElementType = MET_UCHAR\n')
		f.write('ElementDataFile = LOCAL\n')
		
		## Write matrix
		f.write(stru['raw'])
				
		f.close()
		
	elif not fn.endswith('.mha'): ## File extension is not ".mha"
		raise NameError('The input file name is not a mha file!')

######################## WRITE MHA - END - #############################

	 
######################## READ MHA - START - ############################


## This function reads a mha file
##
## INPUT PARAMETER:
## fn=file name
##
## RETURNED PARAMETERS:
## stru=data_structure, it includes:
##			raw=3D/4D matrix
##			size=3D/4D matrix dimension
##			spacing=voxel size
##			offset=spatial offset of raw data
##			data_type='short', 'float' or 'uchar'

def read(fn):
	
	## Utility function - START -
	def __cast2int (l):
		for i in range(3):
			if l[i].is_integer():
				l[i]=int(l[i])
		return l
	## Utility function - END -
	
	stru={'raw':None, 'size':None, 'spacing':None, 'offset':None, 'data_type':None}
	
	if fn.endswith('.mha'): ## Check if the file extension is ".mha"
		
		f = open(fn,'rb')
		data='img' ## On default the matrix is considered to be an image

		## Read mha header
		for r in range(20):
			
			row=f.readline()
			
			if row.startswith('Offset ='):
				row=row.split('=')[1].strip()
				stru['offset']=__cast2int(map(float, row.split()))
			elif row.startswith('ElementSpacing ='):
				row=row.split('=')[1].strip()
				stru['spacing']=__cast2int(map(float, row.split()))
			elif row.startswith('DimSize ='):
				row=row.split('=')[1].strip()
				stru['size']=map(int, row.split())
			elif row.startswith('ElementNumberOfChannels = 3'):
				data='vf' ## The matrix is a vf
			elif row.startswith('ElementType ='):
				data_type=row.split('=')[1].strip()
			elif row.startswith('ElementDataFile ='):
				break
		
		## Read raw data
		stru['raw']=''.join(f.readlines())
		
		f.close()
		
		## Raw data from string to array
		if data_type == 'MET_SHORT':
			stru['raw']=np.fromstring(stru['raw'], dtype=np.int16)
			stru['data_type'] = 'short'
		elif data_type == 'MET_FLOAT':
			stru['raw']=np.fromstring(stru['raw'], dtype=np.float32)
			stru['data_type'] = 'float'
		elif data_type == 'MET_UCHAR':
			stru['raw']=np.fromstring(stru['raw'], dtype=np.uint8)
			stru['data_type'] = 'uchar'
		
		## Reshape array
		if data == 'img':
			stru['raw']=stru['raw'].reshape(stru['size'][2],stru['size'][1],stru['size'][0]).T
		elif data == 'vf':
			stru['raw']=stru['raw'].reshape(stru['size'][2],stru['size'][1],stru['size'][0],3)
			stru['raw']=__shiftdim(stru['raw'], 3).T
			stru['size']=stru['size']+[3]
		
	elif not fn.endswith('.mha'): ## Extension file is not ".mha". It returns all null values
		raise NameError('The input file is not a mha file!')
	
	return stru

######################### READ MHA - END - #############################


#################### UTILITY FUNCTION - START - ########################
############ PRIVATE UTILITY FUNCTION, NOT FOR PUBLIC USE ##############

def __shiftdim (x, n):
		return x.transpose(np.roll(range(x.ndim), -n))

##################### UTILITY FUNCTION - END - #########################
