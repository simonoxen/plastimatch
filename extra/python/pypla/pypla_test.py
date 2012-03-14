########################################################################
## This is a test file for the Plastimatch Python wrapper
## Usage: python pypla_test.py
## Author: Paolo Zaffino  (p.zaffino@yahoo.it)
## Rev 1
## NOT TESTED ON PYTHON 3
########################################################################


import plastimatch as plm
"""
## EXAMPLE TO WARP AN IMAGE
warp=plm.warp() ## Another way to create this objcet is: warp=plm.warp(log_file="log_warp.txt")
warp.log_file="log_warp.txt" ## Log file is not indispensable
warp.option['input']="fix.mha"
warp.option['output-img']="out.mha"
warp.option['xf']="vf.mha"
warp.run_warp()

## EXAMPLE TO CONVERT AN IMAGE
conv=plm.convert() ## Another way to create this objcet is: conv=plm.convert(log_file="log_warp.txt")
conv.log_file="log_convert.txt" ## Log file is not indispensable
conv.option['input']="dicom_dir"
conv.option['output-img']="img_from_dicom.mha"
conv.run_convert()"""

## EXAMPLE TO REGISTER AN IMAGE
reg=plm.register()## Another way to create this objcet is: reg=plm.register(par_file="par.txt", log_file="log.txt")
reg.par_file="par.txt" ## Parameters file is indispensable, it could be also a inexistent file (it will be created)
reg.log_file="log.txt" ## Log file is not indispensable
reg.add_global_stage() ## This section is needed only if the paramaters file does not exist - START - 
reg.stages[0]["fixed"]="fix.mha"
reg.stages[0]["moving"]="mov.mha"
reg.stages[0]["img_out"]="out_img.mha"
reg.add_stage()
reg.stages[1]["xform"]="rigid"
reg.stages[1]["optim"]="versor"
reg.stages[1]["impl"]="itk"
reg.stages[1]["metric"]="mse"
reg.stages[1]["max_its"]="100"
reg.stages[1]["convergence_tol"]="3"
reg.stages[1]["grad_tol"]="1"
reg.stages[1]["res"]="2 2 2"
reg.add_stage()
reg.stages[2]["xform"]="bspline"
reg.stages[2]["optim"]="lbfgsb"
reg.stages[2]["impl"]="plastimatch"
reg.stages[2]["metric"]="mse"
reg.stages[2]["max_its"]="100"
reg.stages[2]["grid_spac"]="33 35 35"
reg.stages[2]["res"]="1 1 1"
reg.add_stage()
reg.stages[3]["xform"]="bspline"
reg.stages[3]["optim"]="lbfgsb"
reg.stages[3]["impl"]="plastimatch"
reg.stages[3]["metric"]="mse"
reg.stages[3]["max_its"]="100"
reg.stages[3]["grid_spac"]="10 10 10"
reg.stages[3]["res"]="1 1 1" ## This section is needed only if the paramaters file does not exist - END -
reg.delete_stage(3) ## Deletes the last stage
reg.run_registration()
