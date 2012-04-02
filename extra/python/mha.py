########################################################################
## This class reads and writes mha files (images or vector fields)
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## Rev 16
## NOT TESTED ON PYTHON 3
########################################################################


import numpy as np


class new():
	
	## PUBLIC PARAMETERS:
	## 
	##	data=3D/4D matrix
	##	size=3D/4D matrix size
	##	spacing=voxel size
	##	offset=spatial offset of data data
	##	data_type='short', 'float' or 'uchar'
	##
	## 
	## CONSTRUCTOR OVERLOADING:
	##
	## img=mha.new() # All the public parameters will be set to None
	## img=mha.new(input_file='img.mha')
	## img=mha.new(data=matrix, size=[512, 512, 80], spacing=[0.9, 0.9, 5], offset=[-240, -240, -160], data_type='short')
	##
	##
	## PUBLIC METHODS:
	##
	## img.read('file_name.mha')
	## img.write('file_name.mha')
	
	data=None
	size=None
	spacing=None
	offset=None
	data_type=None
	
	
######################## CONSTRUCTOR - START - #########################
	
	def __init__ (self, input_file=None, data=None, size=None, spacing=None, offset=None, data_type=None):
		
		if input_file!=None and data==None and size==None and spacing==None and offset==None and data_type==None:
			self.read_mha(input_file)
			
		elif input_file==None and data!=None and size!=None and spacing!=None and offset!=None and data_type!=None:
			self.data=data
			self.size=size
			self.spacing=spacing
			self.offset=offset
			self.data_type=data_type
		
		elif input_file==None and data==None and size==None and spacing==None and offset==None and data_type==None:
			pass
	
######################## CONSTRUCTOR - END - ###########################
	
	
######################## READ_MHA - START - ############################
	
## INPUT PARAMETER:
## fn=file name
##
## This method reads a mha file and assigns the data to the object parameters

	def read_mha(self, fn):
	
		######## Utility function, NOT FOR PUBLIC USE - START - ########
		def __cast2int (l):
			for i in range(3):
				if l[i].is_integer():
					l[i]=int(l[i])
			return l
			
			
		def __shiftdim (x, n):
			return x.transpose(np.roll(range(x.ndim), -n))
		######## Utility function, NOT FOR PUBLIC USE - END - ##########
		
		
		if fn.endswith('.mha'): ## Check if the file extension is ".mha"
			
			f = open(fn,'rb')
			data='img' ## On default the matrix is considered to be an image
	
			## Read mha header
			for r in range(20):
				
				row=f.readline()
				
				if row.startswith('Offset ='):
					row=row.split('=')[1].strip()
					self.offset=__cast2int(map(float, row.split()))
				elif row.startswith('ElementSpacing ='):
					row=row.split('=')[1].strip()
					self.spacing=__cast2int(map(float, row.split()))
				elif row.startswith('DimSize ='):
					row=row.split('=')[1].strip()
					self.size=map(int, row.split())
				elif row.startswith('ElementNumberOfChannels = 3'):
					data='vf' ## The matrix is a vf
				elif row.startswith('ElementType ='):
					data_type=row.split('=')[1].strip()
				elif row.startswith('ElementDataFile ='):
					break
			
			## Read raw data
			self.data=''.join(f.readlines())
			
			f.close()
			
			## Raw data from string to array
			if data_type == 'MET_SHORT':
				self.data=np.fromstring(self.data, dtype=np.int16)
				self.data_type = 'short'
			elif data_type == 'MET_FLOAT':
				self.data=np.fromstring(self.data, dtype=np.float32)
				self.data_type = 'float'
			elif data_type == 'MET_UCHAR':
				self.data=np.fromstring(self.data, dtype=np.uint8)
				self.data_type = 'uchar'
			
			## Reshape array
			if data == 'img':
				self.data=self.data.reshape(self.size[2],self.size[1],self.size[0]).T
			elif data == 'vf':
				self.data=self.data.reshape(self.size[2],self.size[1],self.size[0],3)
				self.data=__shiftdim(self.data, 3).T
				self.size+=[3]
			
		elif not fn.endswith('.mha'): ## Extension file is not ".mha". It returns all null values
			raise NameError('The input file is not a mha file!')
		
######################### READ_MHA - END - #############################
	
	
######################## WRITE_MHA - START - ###########################
	
## INPUT PARAMETER:
## fn=file name
##
## This method writes the object parameters in a mha file
	
	def write_mha (self,fn):
		
		
		if fn.endswith('.mha'): ## Check if the file extension is ".mha"
			
			## Check if the input matrix is an image or a vf
			if self.data.ndim == 3:
				data='img'
			elif self.data.ndim == 4:
				data='vf'
			
			f=open(fn, 'wb')
			
			## Write mha header
			f.write('ObjectType = Image\n')
			f.write('NDims = 3\n')
			f.write('BinaryData = True\n')
			f.write('BinaryDataByteOrderMSB = False\n')
			f.write('CompressedData = False\n')
			f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
			f.write('Offset = '+str(self.offset[0])+' '+str(self.offset[1])+' '+str(self.offset[2])+'\n')
			f.write('CenterOfRotation = 0 0 0\n')
			f.write('AnatomicalOrientation = RAI\n')
			f.write('ElementSpacing = '+str(self.spacing[0])+' '+str(self.spacing[1])+' '+str(self.spacing[2])+'\n')
			f.write('DimSize = '+str(self.data.shape[0])+' '+str(self.data.shape[1])+ ' '+str(self.data.shape[2])+'\n')
			if data == 'vf':
				f.write('ElementNumberOfChannels = 3\n')
				self.data=_shiftdim(self.data, 3) ## Shift dimensions if the input matrix is a vf
			if self.data_type == 'short':
				f.write('ElementType = MET_SHORT\n')
			elif self.data_type == 'float':
				f.write('ElementType = MET_FLOAT\n')
			elif self.data_type == 'uchar':
				f.write('ElementType = MET_UCHAR\n')
			f.write('ElementDataFile = LOCAL\n')
			
			## Write matrix
			f.write(self.data)
			
			f.close()
			
		elif not fn.endswith('.mha'): ## File extension is not ".mha"
			raise NameError('The input file name is not a mha file!')
	
######################## WRITE_MHA - END - #############################
