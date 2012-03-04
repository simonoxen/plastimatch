--  Author: James Shackleford
-- Created: Feb. 29th, 2012
-- Updated: Mar.  4th, 2012

-------------------------------------
--            TUTORIAL             --
-- Working with the Register class --
-- ----------------------------------

-- if we want to register two files, we must first load them
-- here, we will be registering the "moving image" (img_01.mha)
-- to the "fixed image" (img_03.mha)
fimg = Image.load ("data/img_03.mha")
mimg = Image.load ("data/img_01.mha")

-- we can setup a registration by loading registration stage
-- parameters from a standard plastimatch command file
r = Register.load ("data/stages.txt")

-- we must attach the fixed and moving images to the registration
r.fixed = fimg
r.moving = mimg

-- perform the registration. the result is a transform
xf = r:go()

-- of course, we can use this transform to warp the moving image
warp = mimg + xf

-- which we can save
warp:save ("my_warp.mha")

-- now, let's get ready for something more complex
mimg = nil
warp = nil
xf = nil
collectgarbage()


-- let's do a 4D registration

-- first, build an associative array of inputs and outputs
phases = {
    { moving = Image.load ("data/img_01.mha"), warp = "warp_01.mha" },
    { moving = Image.load ("data/img_02.mha"), warp = "warp_02.mha" },
    { moving = Image.load ("data/img_04.mha"), warp = "warp_04.mha" },
    { moving = Image.load ("data/img_05.mha"), warp = "warp_05.mha" }
}

-- let's loop through every input/output pair in our phases array
--    p is the current phase
for _,p in pairs(phases) do
    r.moving = p.moving
    xf = r:go()
    warp = p.moving + xf
    warp:save (p.warp);
end

print ("Tutorial: Register Class -- Completed")
print ("-------------------------------------")
