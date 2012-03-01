-- James Shackleford
-- Feb. 29th, 2012
-------------------------
-- Examples of classes and their functionality
-- as I develop them.  This serves as both
-- documentation and a testbed.
--------------------------------------------------

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

-- multiplying 2 volumes is illegal and will
-- return (nil) & print a warning to stderr
foo = w1*w2
print ("foo:", foo)

-- you can save memory by deleting volumes
-- that are no longer needed. just set them
-- to (nil) and call collectgarbage()
w1 = nil
w2 = nil

collectgarbage()
