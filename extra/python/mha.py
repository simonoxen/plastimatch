## These functions read and write mha files (ONLY IMAGES)
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## rev 10
## NOT TESTED ON PYTHON 3

import numpy as np

def write (A, fn, spacing, offset, data_type):
	
	# A=3D matrix
	# fn=filename
	# data_type='short', 'float' or 'uchar'

	f=open(fn, 'w')
	f.write('ObjectType = Image\n')
	f.write('NDims = 3\n')
	f.write('BinaryData = True\n')
	f.write('BinaryDataByteOrderMSB = False\n')
	f.write('CompressedData = False\n')
	f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
	off='Offset = '+str(offset[0])+' '+str(offset[1])+' '+str(offset[2])+'\n'
	f.write(off)
	f.write('CenterOfRotation = 0 0 0\n')
	f.write('AnatomicalOrientation = RAI\n')
	spac='ElementSpacing = '+str(spacing[0])+' '+str(spacing[1])+' '+str(spacing[2])+'\n'
	f.write(spac)
	dim='DimSize = '+str(A.shape[0])+' '+str(A.shape[1])+ ' '+str(A.shape[2])+'\n'
	f.write(dim)
	if data_type == 'short':
		f.write('ElementType = MET_SHORT\n')
	elif data_type == 'float':
		f.write('ElementType = MET_FLOAT\n')
	elif data_type == 'uchar':
		f.write('ElementType = MET_UCHAR\n')
	f.write('ElementDataFile = LOCAL\n')
	
	f.write(A)
	
	f.close()


def read(fn):
	
	#fn=filename
	
	def cast2int (l):
		for i in range(3):
			if l[i].is_integer():
				l[i]=int(l[i])
		return l
	
	f = open(fn,'r')

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
		elif row.startswith('ElementType ='):
			data_type=row[14:-1]
		elif row.startswith('ElementDataFile ='):
			break
	
	img=f.readlines()
	f.close()
	
	raw_img=''.join(img)
	
	if data_type == 'MET_SHORT':
		raw=np.fromstring(raw_img, dtype=np.int16)
		data_type = 'short'
	elif data_type == 'MET_FLOAT':
		raw=np.fromstring(raw_img, dtype=np.float32)
		data_type = 'float'
	elif data_type == 'MET_UCHAR':
		raw=np.fromstring(raw_img, dtype=np.uint8)
		data_type = 'uchar'
	
	raw=raw.reshape(siz[2],siz[1],siz[0]).T
	
	return (raw, siz, spacing, offset, data_type)
	
	# raw=3D matrix
	# siz=size
	# data_type = 'short', 'float' or 'uchar'
