## These functions read and write mha files (images or vector fields)
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## rev 11
## NOT TESTED ON PYTHON 3

import numpy as np


def write (A, fn, spacing, offset, data_type):
	
	# A=3D/4D matrix
	# fn=file name
	# data_type='short', 'float' or 'uchar'
	
	if fn.endswith('.mha'):
		
		if A.ndim == 3:
			data='img'
		elif A.ndim == 4:
			data='vf'
			
		f=open(fn, 'w')
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
			A=_shiftdim(A, 3)
		if data_type == 'short':
			f.write('ElementType = MET_SHORT\n')
		elif data_type == 'float':
			f.write('ElementType = MET_FLOAT\n')
		elif data_type == 'uchar':
			f.write('ElementType = MET_UCHAR\n')
		f.write('ElementDataFile = LOCAL\n')
		
		f.write(np.array(A))
				
		f.close()
		
	else:
		print ('The input file name is not a mha file!')
	 


def read(fn):
	
	#fn=file name
	
	def cast2int (l):
		for i in range(3):
			if l[i].is_integer():
				l[i]=int(l[i])
		return l

	
	if fn.endswith('.mha'):
		
		f = open(fn,'r')
		data='img'

		for r in range(20):
	
			row=f.readline()
	
			if row.startswith('Offset ='):
				row=row[9:]
				offset=cast2int(map(float, row.split()))
			elif row.startswith('ElementSpacing ='):
				row=row[17:]
				spacing=cast2int(map(float, row.split()))
			elif row.startswith('DimSize ='):
				row=row[10:]
				siz=map(int, row.split())
			elif row.startswith('ElementNumberOfChannels = 3'):
				data='vf'
			elif row.startswith('ElementType ='):
				data_type=row[14:-1]
			elif row.startswith('ElementDataFile ='):
				break
				
		row=''.join(f.readlines())
		
		f.close()
	
		if data_type == 'MET_SHORT':
			raw=np.fromstring(row, dtype=np.int16)
			data_type = 'short'
		elif data_type == 'MET_FLOAT':
			raw=np.fromstring(row, dtype=np.float32)
			data_type = 'float'
		elif data_type == 'MET_UCHAR':
			raw=np.fromstring(row, dtype=np.uint8)
			data_type = 'uchar'
		
		del row
	
		if data == 'img':
			raw=raw.reshape(siz[2],siz[1],siz[0]).T
		elif data == 'vf':
			raw=raw.reshape(siz[2],siz[1],siz[0],3)
			raw=_shiftdim(raw, 3).T
			siz=siz+[3]
		
		return (raw, siz, spacing, offset, data_type)
		
	else:
		print ('The input file is not a mha file!')

	# raw=3D/4D matrix
	# siz=size
	# data_type = 'short', 'float' or 'uchar'



def _shiftdim (x, n):
		return x.transpose(np.roll(range(x.ndim), -n))
