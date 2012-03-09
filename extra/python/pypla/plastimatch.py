########################################################################
## This is a Python wrapper for Plastimatch
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## Rev 1
## NOT TESTED ON PYTHON 3
########################################################################

import os
import subprocess

######################### CLASSES - START - ############################

print ("WARNING: Plastimatch Python wrapper is still in alpha version!")

class warp:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
		
	option={}
		
	warp_keys=("input","default-val","dif","dim","fixed",\
	"input-cxt","input-dose-ast","input-dose-img","input-dose-mc",\
	"input-dose-xio","input-ss-img","input-ss-list","interpolation",\
	"metadata","origin","output-color_map","output-ctx","output-dicom",\
	"output-dij","output-dose_img","output-img","output-labelmap",\
	"output-pointset","output-prefix","output-prefix_fcsv","output-ss_img",\
	"output-ss_list","output-type","output-vf","output-xio","patient-id",\
	"patient-name","patient-pos","prune-empty","referenced-ct","simplify-perc",\
	"spacing","xf","xor-contours")
	
	def _scan_options (self):
		self.option=dict((k, v) for k, v in self.option.iteritems() if k in self.warp_keys)
		opt_str=""
		for key, value in dict.items(self.option):
			opt_str+=" --"+key+"="+value
		return opt_str
		
	def run_warp(self):
		if self.log_file == "":
			self.log=open(os.devnull, "w")
		else:
			self.log=open(self.log_file, "w")
		
		subprocess.call('plastimatch warp' + self._scan_options(), shell=True, stdout=self.log, stderr=self.log)
		
		self.log.close()
		

class convert:
	
	def __init__ (self, log_file=""):
		self.log_file=log_file
		
	option={}
		
	convert_keys=("input","default-val","dif","dim","fixed",\
	"input-cxt","input-dose-ast","input-dose-img","input-dose-mc",\
	"input-dose-xio","input-ss-img","input-ss-list","interpolation",\
	"metadata","origin","output-color_map","output-ctx","output-dicom",\
	"output-dij","output-dose_img","output-img","output-labelmap",\
	"output-pointset","output-prefix","output-prefix_fcsv","output-ss_img",\
	"output-ss_list","output-type","output-vf","output-xio","patient-id",\
	"patient-name","patient-pos","prune-empty","referenced-ct","simplify-perc",\
	"spacing","xf","xor-contours")
	
	def _scan_options (self):
		self.option=dict((k, v) for k, v in self.option.iteritems() if k in self.convert_keys)
		opt_str=""
		for key, value in dict.items(self.option):
			opt_str+=" --"+key+"="+value	
		return opt_str
		
	def run_convert(self):
		if self.log_file == "":
			self.log=open(os.devnull, "w")
		else:
			self.log=open(self.log_file, "w")
		
		subprocess.call('plastimatch convert' + self._scan_options(), shell=True, stdout=self.log, stderr=self.log)
		
		self.log.close()


class register ():
	
	def __init__ (self, par_file="", log_file=""):
		self.par_file=par_file
		self.log_file=log_file
	
	_stage_keys=("xform","optim","impl","background_val","convergence_tol",\
	"demons_acceleration","demons_filter_width","demons_homogenization","demons_std",\
	"histoeq","grad_tol","grid_spac","max_its","max_step","metric","mi_histogram_bins",\
	"min_its","min_step","num_saples","regularization_lambda","res","ss","ss_fixed",\
	"ss_moving","threading","xform_in","xform_out","vf_out","img_out","img_out_fmt","img_out_type")
		
	_global_keys=("fixed","moving","xform_in","xform_out","vf_out","img_out",\
	"img_out_fmt","img_out_type","background_max")

	stages=[]
	_global_stage_added=False
	
	def add_global_stage(self):
		if self.par_file=="" or os.path.exists(self.par_file):
			print ("WARNING:GLOBAL STAGE NOT ADDED! You have to define a new parameters file name")
		else:
			if self._global_stage_added==False:
				self.stages=[{}]+self.stages
				self._global_stage_added=True
			else:
				print ("The global stage already exists!")
		
	def add_stage(self):
		if self.par_file=="" or os.path.exists(self.par_file):
			print ("WARNING:STAGE NOT ADDED! You have to define a new parameters file name")
		else:
			self.stages+=[{}]
	
	def delete_stage(self, stage_number):
		if self.par_file=="" or os.path.exists(self.par_file):
			print ("WARNING:STAGE NOT DELETED! You have to define a new parameters file name")
		else:
			if stage_number != 0:
				del self.stages[stage_number]
			else:
				print ("WARNING: GLOBAL STAGE NOT DELETED! You can not delete the global stage")
	
	def run_registration(self):
		if not os.path.exists(self.par_file) and self.par_file!="":
			f=open(self.par_file, "w")			
			for stage_index, stage in enumerate(self.stages):
				if stage_index==0:
					stage=dict((k, v) for k, v in stage.iteritems() if k in self._global_keys)
					f.write("[GLOBAL]\n")
				else:
					stage=dict((k, v) for k, v in stage.iteritems() if k in self._stage_keys)
					f.write("\n[STAGE]\n")
				
				for key, value in dict.items(stage):
						f.write(key+"="+value+"\n")	
			f.close()
		
		if self.log_file == "":
			self.log=open(os.devnull, "w")
		else:
			self.log=open(self.log_file, "w")
		
		if self.par_file!="" and os.path.exists(self.par_file):
			print ("Please wait...")
			subprocess.call('plastimatch register ' + self.par_file, shell=True, stdout=self.log, stderr=self.log)
			self.log.close()
		else:
			print ("WARNING:REGISTRATION NOT EXECUTED! You have to define a new parameters file name")

########################## CLASSES - END - #############################
