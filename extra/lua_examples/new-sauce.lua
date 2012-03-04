--  Author: James Shackleford
-- Created: Feb. 29th, 2012
-- Updated: Mar.  4th, 2012
-------------------------
-- Examples of classes and their functionality as I develop them.
-- This serves as both documentation and a testbed.
----------------------------------------------------------------------

--------------------
-- Act 1 : Images --
--------------------

-- load an image
my_image = image.load ("test.mha")

-- "Save" (write it to disk)
my_image:save ()

-- "Save-As" (write it to disk as a new file)
my_image:save ("test-clone.mha")

-- scale total volume intensity by a constant
w1 = 0.1*my_image
w1:save ("w1.mha");

-- the order doesn't matter
w2 = my_image*0.9
w2:save ("w2.mha");

-- multiplying 2 volumes is illegal and will return (nil) & print
-- a warning to stderr.
foo = w1*w2
print ("foo:", foo)

-- derived images cannot be saved without explicitly specifying a
-- filename.  this will generate an error to stderr.
w3 = 2.5*w1
w3:save ();

-- but once a filename is specifed, you can save without specifying
-- a name.  the last explicitly specified filename sticks to the image
w3:save ("w3.mha");
w3:save ();

-- you can save memory by deleting volumes that are no longer needed.
-- just set them to (nil) and call collectgarbage()
w1 = nil
w2 = nil
w3 = nil

collectgarbage()


------------------------
-- Act 2 : Transforms --
------------------------

-- you can load a deformation transform like this
xf1 = xform.load ("vf.mha")

-- or like this...
xf2 = xform.load ("xf.txt")

-- and you can warp an image by adding a compatible transform
warp1 = my_image + xf1
warp2 = my_image + xf2

-- and, of course, you can save the warp
warp1:save ("warp1.mha")
warp2:save ("warp2.mha")

-- you cannot add transforms to make composites...yet
-- currently, this will print a warning to stderr and return (nil)
new_xform = xf1 + xf2

xf1 = nil
xf2 = nil
warp1 = nil
warp2 = nil

collectgarbage()


--------------------------
-- Act 3 : Registration --
--------------------------

-- if we want to register two files, we must first load them
fimg = image.load ("fixed.mha")
mimg = image.load ("moving.mha")

-- we can setup a registration by loading registration stage
-- parameters from a standard plastimatch command file
my_reg = register.load ("commandfile.txt")

-- we must attach the fixed and moving images to the registration
my_reg.fixed = fimg
my_reg.moving = mimg

-- perform the registration. the result is a transform, here
-- we save it into xf
xf = my_reg:go()

-- of course, we can use this transform to warp images
my_warp = mimg + xf

-- which we can save
my_warp:save ("my_warp.mha")

-- you can see how powerful this becomes when combined
-- with a simple loop


print ("----------------")
print ("   [THE END]")
print ("----------------")
