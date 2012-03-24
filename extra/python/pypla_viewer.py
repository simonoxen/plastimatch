########################################################################
## This is the default viewer for the pypla project
##
## FUNCTION STILL IN TESTING
##
## Author: Paolo Zaffino (p.zaffino@yahoo.it)
##
## rev 1
##
## Required libraries:
## 1) Numpy
## 2) Matplotlib
## 3) mha.py file
##
## NOT TESTED ON PYTHON 3
##
##
## BRIEF DOCUMENTATION:
##
## input = Input image, it can be a mha file or a mha.read object (from mha.py)
## slice = Slice number, default slice is the middle one
## view = View, default is coronal, choices='a','c' and 's'
## overlay_image = Overlay image, it can be a mha file or a mha.read object (from mha.py)
## gain_overlay_image = Gain of overlay image
## windowing = Windowing interval as "-100 100"
## windowing_overlay_img = Windowing interval for the overlay image
## vf = Input vector field in order to plot the phase map, it can be a mha file or a mha.read object (from mha.py)
## checkerboard = Checkerboard mode, to enable set on True
## check_size = Size of the check (in voxel), default is 100
## diff = Shows the difference between the input image and the overlay one. To enable set on True
## colours = Allows the colours in the overlay mode. To enable set on True
## screenshot_file_name = file name where will be saved the screenshot
########################################################################


import matplotlib.cm as cm
from matplotlib.pyplot import imshow, show, hold, savefig
from mha import read
import numpy as np


########################################################################


def show_img(input='', slice=None, view='c', overlay_image=None, gain_overlay_image=1, windowing='', windowing_overlay_img='',\
vf=None, checkerboard=False, check_size=100, diff=False, colours=False, screenshot_file_name=None):
	
	## Read the input image
	if type(input)==str:
		img=read(input)
	elif type(input)==dict:
		img=input
		
	## Read the vector field
	if vf != None:
		
		if type(vf)==str:
			vf_data=read(vf)
		elif type(vf)==dict:
			vf_data=vf
	
	## Slice number settings
	if slice == None and (view == 'c' or view == 's'):
		slice_number=np.rint(img['size'][1]/2)
	elif slice == None and view == 'a':
		slice_number=np.rint(img['size'][2]/2)
	else:
		slice_number=slice
	
	
	## View settings
	if view == 'c':
		slice=img['raw'][:,slice_number].T
		pixel_ratio=img['spacing'][2]/img['spacing'][0]
		if vf != None:
			phase_vf=np.arctan(vf_data['raw'][:,slice_number,:,0], vf_data['raw'][:,slice_number,:,2]).T
			
	elif view == 's':
		slice=img['raw'][slice_number,:].T
		pixel_ratio=img['spacing'][2]/['img_spacing'][1]
		if vf != None:
			phase_vf=np.arctan(vf_data['raw'][slice_number,:,:,1], vf_data['raw'][slice_number,:,:,2]).T
			
	elif view == 'a':
		slice=np.rot90(img['raw'][:,:,slice_number],1)
		pixel_ratio=img['spacing'][1]/['img_spacing'][0]
		if vf != None:
			phase_vf=np.rot90(np.arctan(vf_data['raw'][:,:,slice_number,0], vf_data['raw'][:,:,slice_number,1]), 1)
		
	if type(vf)==str:
		del vf_data['raw']
	
	if type(input)==str:
		del img['raw']
	
	
	## Overlay settings
	if overlay_image != None:
		
		if type(overlay_image)==str:
			img2=read(overlay_image)
		elif type(overlay_image)==dict:
			img2=overlay_image
			
		if img['size'] != img2['size']:
			print "Warning: the two images don't have the same dimensions!"
		if img['spacing'] != img2['spacing']:
			print "Warning: the two images don't have the same pixel spacing!"
		if img['offset'] != img2['offset']:
			print "Warning: the two images don't have the same offset!"
	
		if view == 'c':
			slice2=img2['raw'][:,slice_number].T
		elif view == 's':
			slice2=img2['raw'][slice_number,:].T
		elif view == 'a':
			slice2=np.rot90(img2['raw'][:,:,slice_number],1)
		
		if type(overlay_image)==str:
			del img2['raw']
	
	
	## Windowing settings
	if windowing != '':
	
		windowing_low_value=int(windowing.split(' ')[0])
		windowing_hi_value=int(windowing.split(' ')[1])
		slice[slice<windowing_low_value]=0
		slice[slice>windowing_hi_value]=0
	
	if windowing_overlay_img != '' and overlay_image != None:
		slice2[slice2<windowing_low_value]=0
		slice2[slice2>windowing_hi_value]=0
	
	
	## Show options
	if diff == False and colours == False and overlay_image == None and vf == None and checkerboard == False: ## One image, NO options
		imshow(slice, cmap=cm.gray, aspect=pixel_ratio, origin='lower')
	
	elif diff == False and colours == True and overlay_image != None and vf == None and checkerboard == False: ## Overlay images in colours mode, NO checkerboard, NO vf
		imshow(slice, cmap=cm.gray, aspect=pixel_ratio, origin='lower')
		imshow(slice2, aspect=pixel_ratio, origin='lower', alpha=0.5)
	
	elif diff == False and colours == False and overlay_image != None and vf == None and checkerboard == False: ## Overlay images, NO checkerboard, NO colours, NO vf
		slice_sum=np.add(np.multiply(slice2, gain_overlay_image), slice)
		del slice, slice2
		imshow(slice_sum, cmap=cm.gray, aspect=pixel_ratio, origin='lower')
	
	elif diff == False and overlay_image != None and checkerboard == True: ## Overlay images in checkerboard mode
		check_white=np.ones((np.rint(check_size/pixel_ratio), check_size))
		check_black=np.zeros((np.rint(check_size/pixel_ratio), check_size))
		check_number_x=slice.shape[1]/check_size
		check_number_y=slice.shape[0]/np.rint(check_size/pixel_ratio)
		
		x=y=0
		while x <= check_number_x:
			if x == 0:
				row=np.concatenate((check_white, check_black), axis=1)
				row_neg=np.concatenate((check_black, check_white), axis=1)
			else:
				row=np.concatenate((row, check_white), axis=1)
				row=np.concatenate((row, check_black), axis=1)
			x=x+2
		
		row_neg=np.ones(row.shape)
		row_neg=np.subtract(row_neg, row)
		
		while y <= check_number_y:
			if y == 0:
				checkerboard=np.concatenate((row, row_neg), axis=0)
			else:
				checkerboard=np.concatenate((checkerboard, row), axis=0)
				checkerboard=np.concatenate((checkerboard, row_neg), axis=0)
			y=y+2
		
		checkerboard=np.delete(checkerboard, np.s_[slice.shape[0]:checkerboard.shape[0]], axis=0)
		checkerboard=np.delete(checkerboard, np.s_[slice.shape[1]:checkerboard.shape[1]], axis=1)
		
		checkerboard_neg=np.ones(checkerboard.shape)
		checkerboard_neg=np.subtract(checkerboard_neg, checkerboard)
		
		slice=np.multiply(slice, checkerboard)
		slice2=np.multiply(slice2, checkerboard_neg)
		
		if colours == True: ## Checkerboad color mode
			imshow(slice, cmap=cm.get_cmap('bone'), aspect=pixel_ratio, origin='lower')
			imshow(slice2, cmap=cm.get_cmap('reds'), aspect=pixel_ratio, origin='lower', alpha=0.70)
		elif colours == False: ## Checkerboad NO color mode
			imshow(slice, cmap=cm.gray, aspect=pixel_ratio, origin='lower')
			imshow(slice2, cmap=cm.gray, aspect=pixel_ratio, origin='lower', alpha=0.50)
			if vf != None: ## Checkerboad NO color mode + vf
				imshow(phase_vf, aspect=pixel_ratio, origin='lower', alpha=0.5)
	
	
	elif diff == False and colours == False and overlay_image == None and vf != None and checkerboard == False: ## Image and vector field
		imshow(slice, cmap=cm.gray, aspect=pixel_ratio, origin='lower')
		imshow(phase_vf, aspect=pixel_ratio, origin='lower', alpha=0.5)
	
	elif diff == True and colours == False and overlay_image != None and vf == None and checkerboard == False: ## Two images in diff mode
		diff_slice=np.subtract(slice, np.multiply(slice2, gain_overlay_image))
		del slice, slice2
		imshow(diff_slice, cmap=cm.gray, aspect=pixel_ratio, origin='lower')
	
	
	## Print the result on the screen or into a file
	if screenshot_file_name==None:
		show()
	else:
		savefig(screenshot_file_name)
		print ("Screenshot saved into the file: " + screenshot_file_name)
	
